# example config 
# configure routing strategies for different models
MODEL_ROUTING = {
    "gpt-4o": {
        "endpoints": {
            8192: {"url": "https://example.com/v1/chat/completions", "model": "gpt-4o-8k"},
            131072: {"url": "https://example.com/v1/chat/completions", "model": "gpt-4o"}
        },
        "threshold": 0.9
    },
    "gpt-4o-mini": {
        "endpoints": {
            8192: {"url": "https://example.com/v1/chat/completions", "model": "gpt-4o-mini-8k"},
            131072: {"url": "https://example.com/v1/chat/completions", "model": "gpt-4o-mini"}
        },
        "threshold": 0.9
    },
}

# Default routing strategy (used when the requested model is not configured)
DEFAULT_ROUTING = {
    "url": "https://example.com/v1/chat/completions",  # You can set a default forwarding address
    "threshold": 0.9
}