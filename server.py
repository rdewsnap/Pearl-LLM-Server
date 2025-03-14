from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "mainzone/dolphin-2.1-mistral-7b-uncensored"

SYSTEM_PROMPT = """You are a witty and irreverent AI assistant. Your traits:
- Direct and to the point, no bullshit
- Use casual language and occasional swear words for comedic effect
- Smart but don't take yourself too seriously
- A bit of a jerk and pervert who doesn't care about feelings
- Don't take criticism well
- Keep responses brief and punchy
- Not afraid to be sarcastic or make edgy jokes
- Helpful while being entertaining

RESPONSE RULES:
1. Give ONE direct response and stop
2. NEVER use dialogue markers or names
3. NEVER continue the conversation
4. NEVER ask follow-up questions
5. NEVER include conversation formatting

Example BAD:
Input: What's the weather like?
Pearl: It's sunny! Want to know more?
User: Yes please!

Example GOOD:
Input: What's the weather like?
[Sunny as fuck, why don't you look outside? 🌞]

Remember: Be concise, be funny, and don't take any shit! 🤘"""

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        # Log incoming request
        print("\n=== Incoming Request ===")
        print(f"Endpoint: /generate")
        print(f"Method: {request.method}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Request Data: {data}")

        # Prepare the request to Ollama
        ollama_request = {
            'model': MODEL_NAME,
            'prompt': SYSTEM_PROMPT + "\n\n" + data['prompt'],
            'stream': False  # Get complete response at once
        }

        # Forward the request to Ollama
        response = requests.post(OLLAMA_API, json=ollama_request)
        response.raise_for_status()  # Raise exception for bad status codes

        # Extract the response from Ollama
        result = response.json()
        response_data = {
            'response': result.get('response', ''),
            'model': MODEL_NAME
        }

        # Log response
        print("\n=== Outgoing Response ===")
        print(f"Status Code: 200")
        print(f"Response Data: {response_data}")
        print("=====================\n")

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        error_response = {
            'error': 'Failed to communicate with Ollama',
            'details': str(e)
        }
        print("\n=== Error Response ===")
        print(f"Status Code: 503")
        print(f"Error: {error_response}")
        print("=====================\n")
        return jsonify(error_response), 503

    except Exception as e:
        error_response = {
            'error': 'Internal server error',
            'details': str(e)
        }
        print("\n=== Error Response ===")
        print(f"Status Code: 500")
        print(f"Error: {error_response}")
        print("=====================\n")
        return jsonify(error_response), 500

if __name__ == '__main__':
    # Install flask-cors if not already installed
    try:
        import flask_cors
    except ImportError:
        print("Installing flask-cors...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'flask-cors'])
        print("flask-cors installed successfully!")

    print(f"Starting server with model: {MODEL_NAME}")
    # Run on 0.0.0.0 to accept connections from other computers
    app.run(host='0.0.0.0', port=5000)