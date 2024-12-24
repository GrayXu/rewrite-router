# token router

An OpenAI-compatible API router that routes based on **context token length**, similar to `moonshot-v1-auto`, routing requests to `moonshot-v1-8k`, `moonshot-v1-32k`, and `moonshot-v1-128k`.


```bash
pip install -r requirements.txt
# And modify `config.py` for your own configuration
python ./run.py --port=8080
```