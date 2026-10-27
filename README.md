# Love Predictor 💕

A playful web app that guesses your chance of finding a relationship, explains what is helping or holding you back, and gives you personal AI-written tips to boost your chances.

> Just for fun! The model is trained on made-up data, so the result is a playful guess, not real relationship advice.

## Features

- **Love forecast:** answer 11 simple questions and a logistic regression model predicts your chance of finding a relationship as a percentage.
- **Insights:** see how you compare with the average person, what share of people have a lower chance than you, and which of your answers raise or lower your chance the most.
- **AI tips:** get three personal, 3-sentence tips written by a small language model, each focused on the change that would help you most.
- **Friendly design:** hand-drawn, doodle-style pages with page transitions, sliders and a description under every question.

## Screenshots

| Answer a few easy questions | Get your forecast, insights and tips |
|---|---|
| ![Question form with a description under every field, sliders for confidence and social activity, and Clear and See my love forecast buttons](docs/screenshots/form.png) | ![Result page showing a 38% chance, a comparison with the average person, what is working and what is holding you back, and three AI-written tips](docs/screenshots/result.png) |

## How it works

1. **Prediction:** a machine learning model (logistic regression), trained on [dataset.csv](dataset.csv), predicts the chance from your answers.
2. **Insights:** the same model explains the result by measuring how much each of your answers raises or lowers your chance.
3. **Recommendations:** the model finds the three changes that would raise your chance the most, and a Hugging Face language model ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)) writes a tip for each one.

## Getting started

### Requirements

- Python 3.10 or newer (built with Python 3.13)
- About 1 GB of free disk space for the AI model, downloaded automatically on first run
- An internet connection on first run (for the model and the Google Fonts)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Then open http://127.0.0.1:5001 in your browser.

The first start takes a little longer because the AI model is downloaded and loaded. Tips take about a minute to generate on a regular CPU, so the forecast and insights show up first and the tips appear when they are ready.

### Run with Docker

You can also build and run the app as a Docker image, so you don't need to install Python or any packages yourself. You only need [Docker](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
docker build -t love-predictor .
docker run -p 5001:5001 love-predictor
```

Then open http://localhost:5001.

- **Build time and size:** the first build takes a few minutes and the image is about 4 GB, because it includes PyTorch and the AI model. Later builds are much faster when only the code changes.
- **Works offline:** the AI model and the trained prediction model are built into the image, so the container runs without an internet connection.
- **Production server:** the image runs the app with [gunicorn](https://gunicorn.org/) instead of Flask's development server. To use a different port, set the `PORT` variable, for example `docker run -e PORT=8080 -p 8080:8080 love-predictor`.
- **Open it on other devices:** the container listens on your whole local network, so phones and laptops on the same Wi-Fi can open `http://<this computer's IP>:5001` (on Windows, find the IP with `ipconfig`; on macOS or Linux, `ipconfig getifaddr en0` or `hostname -I`). If it doesn't load, allow port 5001 through your computer's firewall. To keep it private to your computer, run `docker run -p 127.0.0.1:5001:5001 love-predictor` instead.
- **Stop it:** press Ctrl + C, or run `docker ps` to find the container and `docker stop <container id>`.

### Ready-made image

A [GitHub Actions workflow](.github/workflows/docker.yml) builds the Docker image on every push and pull request to `main`, starts it and tests the running app, including the AI tips. When a push to `main` passes, the image is published to GitHub Container Registry, so you can skip building it:

```bash
docker run -p 5001:5001 ghcr.io/p1tp3t3/relationship-predictor:latest
```

## Project structure

```
love-predictor/
├── main.py                       # Flask app: pages and API routes
├── requirements.txt              # Python packages to install
├── Dockerfile                    # Builds the Docker image (gunicorn, AI model included)
├── .github/workflows/docker.yml  # Builds, tests and publishes the Docker image
├── generate_dataset.py           # Creates the synthetic dataset.csv
├── dataset.csv                   # 1,000 rows of made-up training data
├── docs/screenshots/             # Screenshots used in this README
├── model/
│   ├── love_predictor.py         # Logistic regression model, insights and what-if improvements
│   ├── love_predictor.joblib     # Saved trained model (created automatically)
│   └── recommender.py            # AI-written tips using Hugging Face transformers
└── templates/
    ├── _styles.html              # Shared doodle-style CSS, included in both pages
    ├── index.html                # Question form
    └── result.html               # Forecast, insights and tips
```

## API

All endpoints take and return JSON.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/model/predict` | Returns `prediction` (0 or 1), `probability` and `insights` |
| `POST` | `/api/model/recommend` | Returns up to three AI-written `recommendations`. Responds with `503` if the AI model is not available |
| `GET` | `/api/model/evaluate` | Returns the model's `accuracy` on a held-out test split |

Example request body for `predict` and `recommend`:

```json
{
  "age": 25,
  "sex": 0,
  "past_relationships": 2,
  "years_single": 1.5,
  "social_activity": 5,
  "confidence": 6,
  "hobbies_count": 3,
  "friends_count": 10,
  "uses_dating_apps": 1,
  "goes_out_per_month": 4
}
```

## The dataset

[dataset.csv](dataset.csv) is synthetic. [generate_dataset.py](generate_dataset.py) creates 1,000 people and gives each a hidden chance based on their answers plus random noise. Confidence, social activity and using dating apps raise the chance, while being single for a long time lowers it. Sex has no effect.

| Column | Meaning |
|---|---|
| `age` | 18 to 60 |
| `sex` | 0 = male, 1 = female |
| `past_relationships` | Number of serious past relationships |
| `years_single` | Years since the last relationship |
| `social_activity` | 1 (homebody) to 10 (very social) |
| `confidence` | 1 (shy) to 10 (very confident) |
| `hobbies_count` | Number of regular hobbies |
| `friends_count` | Number of close friends |
| `uses_dating_apps` | 0 = no, 1 = yes |
| `goes_out_per_month` | Times going out to socialize per month |
| `will_have_relationship` | The label: 0 = no, 1 = yes |

To change the number of rows or the hidden formula, edit `generate_dataset.py` and run:

```bash
python generate_dataset.py
```

The current model reaches about 68.5% accuracy on held-out data.

## Customization

- **Better tips:** change `MODEL_NAME` in [model/recommender.py](model/recommender.py) to a bigger model such as `Qwen/Qwen2.5-1.5B-Instruct`. Tips get better but take about three times longer.
- **Tip length:** change `TIP_SENTENCES` and the wording of `SYSTEM_PROMPT` in the same file.
- **Port:** change the port in the last line of [main.py](main.py).
