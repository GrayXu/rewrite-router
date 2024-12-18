from flask import Flask, request, Response, stream_with_context
import requests
import tiktoken
import json

from config import MODEL_ROUTING, DEFAULT_ROUTING

app = Flask(__name__)

# Tokenizer used
TOKENIZER = tiktoken.get_encoding("cl100k_base")

def get_token_count(text):
    """Get the number of tokens in the text"""
    return len(TOKENIZER.encode(text))

def select_endpoint(prompt, model_routing):
    """Select the appropriate endpoint based on the number of tokens and model routing"""
    token_count = get_token_count(prompt)
    print(f"Token count: {token_count}")

    endpoints = model_routing.get("endpoints")
    threshold = model_routing.get("threshold", DEFAULT_ROUTING["threshold"])

    if endpoints:
        for context_length, endpoint in sorted(endpoints.items()):
            if token_count <= context_length * threshold:
                return endpoint
        return endpoints[max(endpoints.keys())]
    else:
        return {"url": DEFAULT_ROUTING["url"], "model": request.get_json().get('model')}

def forward_request(endpoint, request_data, headers):
    """Forward the request to the target URL and handle the response"""
    try:
        response = requests.post(endpoint['url'], json=request_data, headers=headers, stream=True)
        response.raise_for_status()

        def generate():
            for line in response.iter_lines():
                if line:
                    yield f"{line.decode('utf-8')}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except requests.exceptions.RequestException as e:
        error_message = f"Error forwarding request: {e}"
        print(error_message)
        return Response(error_message, status=502)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completion requests"""
    request_data = request.get_json()

    if not request_data or 'messages' not in request_data or not isinstance(request_data['messages'], list):
        return Response("Invalid request: 'messages' field is missing or not a list", status=400)

    # Extract prompt
    prompt = ""
    for message in request_data['messages']:
        prompt += message.get("content", "")

    # Select routing strategy based on the requested model
    requested_model = request_data.get("model")
    model_routing = MODEL_ROUTING.get(requested_model, DEFAULT_ROUTING) # Use default if not configured

    # Select endpoint
    endpoint = select_endpoint(prompt, model_routing)
    print(f"Selected endpoint: {endpoint}")

    # Replace model name (if endpoints are provided in routing strategy and the selected endpoint has a model field)
    if "endpoints" in model_routing and endpoint.get("model"):
        request_data['model'] = endpoint.get("model")

    # Forward request and get response
    headers = {
        'Authorization': request.headers.get('Authorization'),
        'Content-Type': 'application/json'
    }

    return forward_request(endpoint, request_data, headers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)