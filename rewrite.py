from flask import Flask, jsonify, request, Response, stream_with_context
import requests
import json
import tiktoken
from datetime import datetime

app = Flask(__name__)

from config import BACKEND_URL, REWRITE_RULES, ROUTING_RULES

# Tokenizer used
TOKENIZER = tiktoken.get_encoding("cl100k_base")

def get_token_count(text):
    """Get the number of tokens in the text"""
    return len(TOKENIZER.encode(text))

def select_model(prompt, model_routing):
    """Select the appropriate model based on the number of tokens and model routing"""
    token_count = get_token_count(prompt)
    # print(f"Token count: {token_count}")

    models = model_routing["models"]
    threshold = model_routing["threshold"]

    for context_length, model in sorted(models.items()):
        if token_count <= context_length * threshold:
            return model
        
    return models[max(models.keys())]

def forward_request(url, request_data, headers):
    """Forward the request to the target URL and handle the response"""
    try:
        response = requests.post(url, json=request_data, headers=headers, stream=True)
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
def handle_chat_completions():
    # print('-'*20)
    # Get the request body
    data = request.get_json()
    
    if not data or 'messages' not in data or not isinstance(data['messages'], list):
        return Response("Invalid request: 'messages' field is missing or not a list", status=400)
    
    if 'model' not in data:
        return Response("Invalid request: 'model' field is missing", status=400)
    
    # Auto routing
    if data['model'] in ROUTING_RULES:
        # Extract prompt
        prompt = ""
        for message in data['messages']:
            prompt += message.get("content", "")
        model_routing = ROUTING_RULES[data['model']]
        model_selected = select_model(prompt, model_routing)
        print(f"\t\trouting {data['model']} to {model_selected}")
        data['model'] = model_selected
        
    # Rewrite requests
    if data['model'] in REWRITE_RULES:
        print(f"\t\trewrite {data['model']}")
        for rule_key, rule_value in REWRITE_RULES[data['model']].items():
            # Insert prompts
            if rule_key == 'message':
                for message in rule_value:  
                    data['messages'].insert(0, message)
            # Rewrite messages
            elif rule_key == 'tools':
                if 'tools' not in data:
                    data['tools'] = []
                for tool in rule_value:
                    data['tools'].append(tool)
            # Rewrite parameters
            else:
                data[rule_key] = rule_value
    
    # print(data)  # After processing
    
    headers = {
        'Authorization': request.headers.get('Authorization'),  # Pass through the authorization header
        'Content-Type': 'application/json'
    }

    return forward_request(
        f"{BACKEND_URL}/v1/chat/completions",
        data,
        headers
    )

@app.route('/v1/models', methods=['GET','POST'])
def list_models():
    # print('-'*10+'list_models'+'-'*10)
    # Forward the request to the backend
    headers = {
        'Authorization': request.headers.get('Authorization'),  # Pass through the authorization header
        'Content-Type': 'application/json'
    }
    response = requests.get(  # Change to GET method
        f"{BACKEND_URL}/v1/models", 
        headers=headers,
        stream=False
    )
    # print(response.text)
    response_data = response.json()

    if 'data' not in response_data or not isinstance(response_data['data'], list):
        return Response("Invalid response from backend", status=500)

    model_list = response_data['data']  # Use the model list returned by the backend as the base

    # Get the created timestamp (if the list is not empty)
    created_timestamp = 1677649963  # Default timestamp
    if model_list:
        created_timestamp = model_list[0]['created']

    # Add models from REWRITE_RULES to the list
    existing_ids = {model['id'] for model in model_list}
    for model_name in REWRITE_RULES.keys():
        if model_name not in existing_ids:
            model_list.append({
                "id": model_name,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "user"
            })

    return jsonify({
        "data": model_list,
        "object": "list"
    })

# Proxy others
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy_others(path):
    # print(f"Proxying {path}")
    url = f"{BACKEND_URL}/{path}"

    headers = {key: value for key, value in request.headers.items() if key.lower() != 'host'}
    method = request.method.lower()

    # Handle request body and query parameters
    data = request.get_data() if request.method in ['POST', 'PUT', 'PATCH'] else None
    params = request.args if request.method == 'GET' else None

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            params=params,
            stream=True
        )
        response.raise_for_status()

        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        return Response(
            stream_with_context(generate()),
            content_type=response.headers.get('Content-Type'),
            status=response.status_code
        )
    except requests.exceptions.RequestException as e:
        error_message = f"Error forwarding request: {e}"
        print(error_message)
        return Response(error_message, status=502)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run rewrite router with specified host and port.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='run on (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=3034, help='Port to run on (default: 3034)')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)