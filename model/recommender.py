import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small instruction-tuned chat model that runs on CPU. Swap for a bigger one
# (e.g. 'Qwen/Qwen2.5-1.5B-Instruct') for better tips at the cost of speed.
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'

SYSTEM_PROMPT = (
    'You are a warm, encouraging dating coach talking directly to the user, addressing them as "you". '
    'Reply with one paragraph of exactly three sentences: first give one concrete thing to do this week, '
    'then explain how to do it, and end by saying why it helps with finding a relationship. '
    'Use simple, friendly words. No lists, no headings, no line breaks, no preamble.'
)

TIP_SENTENCES = 3
MIN_TIP_WORDS = 15
# How many times to ask the model for a tip before giving up on it
MAX_ATTEMPTS = 3


class RecommenderUnavailable(RuntimeError):
    # Raised when the transformer model could not be loaded or failed to generate
    pass

# First-person descriptions used to build the prompt for each feature
FEATURE_PROMPTS = {
    'social_activity': 'I rate my social activity {current:g} out of 10',
    'confidence': 'I rate my self-confidence {current:g} out of 10',
    'hobbies_count': 'I only have {current:g} hobbies',
    'friends_count': 'I only have {current:g} friends',
    'uses_dating_apps': "I don't use dating apps",
    'goes_out_per_month': 'I only go out about {current:g} times a month',
}

# What the tip must be about, so the small model stays on topic
FEATURE_GOALS = {
    'social_activity': 'being more social and spending more time around people',
    'confidence': 'feeling more confident when meeting new people',
    'hobbies_count': 'picking up a new hobby where I can meet people',
    'friends_count': 'making new friends and growing my friend group',
    'uses_dating_apps': 'getting started with a dating app like Hinge or Bumble',
    'goes_out_per_month': 'going out more often to places where I can meet people',
}

FEATURE_LABELS = {
    'social_activity': 'Social activity',
    'confidence': 'Confidence',
    'hobbies_count': 'Hobbies',
    'friends_count': 'Friends',
    'uses_dating_apps': 'Dating apps',
    'goes_out_per_month': 'Going out',
}


class Recommender:
    # Turns the predictor's what-if improvements into personal tips using a Hugging Face transformer.
    # The model is loaded once when the app starts, because it is large and slow to load.
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_error = None
        self._lock = threading.Lock()
        self._load()  # Load the model on initialization to avoid delays on first request

    def _load(self):
        with self._lock:
            if self.model is not None or self.load_error is not None:
                return
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Decoder-only models must be padded on the left for batched generation
                self.tokenizer.padding_side = 'left'
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.eval()
            except Exception as e:
                self.load_error = str(e)

    def _build_prompt(self, item, age):
        situation = FEATURE_PROMPTS[item['feature']].format(current=item['current'])
        goal = FEATURE_GOALS[item['feature']]
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': (
                f"I'm {age:g} and want to find a relationship. {situation}. "
                f'Give me advice only about {goal}. '
                'What is one specific thing I can do this week? Answer in one paragraph of three sentences.'
            )},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=130,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Drop the prompt tokens so only the model's reply is decoded
        replies = outputs[:, inputs['input_ids'].shape[1]:]
        return [self._clean(text) for text in self.tokenizer.batch_decode(replies, skip_special_tokens=True)]

    @staticmethod
    def _clean(text):
        # Squash lists and line breaks into one paragraph, keep at most TIP_SENTENCES
        # sentences and drop a final sentence that was cut off mid-way
        text = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*|__|#+\s*', '', text)
        text = ' '.join(text.split())
        sentences = re.findall(r'[^.!?]+[.!?]+["\')]?', text)
        return ' '.join(sentence.strip() for sentence in sentences[:TIP_SENTENCES])

    def recommend(self, improvements, age):
        if not improvements:
            return {'source': 'none', 'recommendations': []}

        if self.model is None:
            raise RecommenderUnavailable(self.load_error or 'Model is not loaded')

        prompts = [self._build_prompt(item, float(age)) for item in improvements]
        tips = [''] * len(prompts)
        try:
            # Small models sometimes return an empty or very short answer,
            # so ask again for just those tips
            for _ in range(MAX_ATTEMPTS):
                retry = [i for i, tip in enumerate(tips) if len(tip.split()) < MIN_TIP_WORDS]
                if not retry:
                    break
                for i, tip in zip(retry, self._generate([prompts[i] for i in retry])):
                    if len(tip.split()) > len(tips[i].split()):
                        tips[i] = tip
        except Exception as e:
            raise RecommenderUnavailable(str(e)) from e

        return {
            'source': 'transformer',
            # Leave out any tip the model never managed to write
            'recommendations': [
                {
                    'feature': item['feature'],
                    'label': FEATURE_LABELS[item['feature']],
                    'current': item['current'],
                    'target': item['target'],
                    'gain': item['gain'],
                    'tip': tip,
                }
                for item, tip in zip(improvements, tips)
                if tip
            ],
        }
