# token router

An OpenAI-compatible API router that routes based on **context token length**, similar to `moonshot-v1-auto`, routing requests to `moonshot-v1-8k`, `moonshot-v1-32k`, and `moonshot-v1-128k`.

You can configure rules for different token lengths, endpoints, and models in `config.py`, and token override is also supported.

```bash
pip install -r requirements.txt

# And modify `config.py` for your own configuration

python ./run.py --port=8080

# or with gunicorn (pip install gunicorn)

gunicorn --workers 2 --threads 2 --bind 0.0.0.0:3033 run:app
```

ps: rewrite.py is used to rewrite parameters in the request data. (e.g. specifying max_tokens for 01.ai as 5000)