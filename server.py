from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re
import json
from config import SERPER_API_KEY
import random
from typing import List, Optional, Dict, Any, Sequence

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "dolphin-mistral"
SERPER_API = "https://google.serper.dev/search"

# Message and Context Configuration
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class ConversationManager:
    def __init__(self, max_messages: int = 8):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.context_tokens: Optional[Sequence[int]] = None
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.messages.append(Message(role, content))
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
            
    def get_formatted_context(self) -> str:
        """Get formatted context string for prompt construction"""
        formatted = []
        for msg in self.messages[-2:]:  # Only use last 2 messages for prompt context
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n---\n".join(formatted) if formatted else ""
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.messages.clear()
        self.context_tokens = None

# Initialize conversation manager
conversation = ConversationManager()

# Search Related Constants
PRICE_RELATED_TERMS = ['price', 'btc', 'bitcoin', 'eth', 'ethereum', 'stock']
SEARCH_PREFIX = "Search: "

# Follow-up Question Indicators
FOLLOWUP_INDICATORS = [
    'what did', 'previous', 'before', 'earlier', 'last time',
    'what', 'which', 'who', 'where', 'when', 'why', 'how'
]

# Response Cleaning Patterns
FORMAT_MARKERS = [
    'Current message:', 'Previous messages:', 'Response:', 'Pearl:', '###',
    "Well, you asked for it, so here's the deal:", "Here's what I found:",
    "Current price:", "Price:", "Let's perform a search", "According to",
    "That's it for now", "The current", "Current", "I suggest",
    "You could try", "You might want to", "For example", "Here are some",
    "Such as", "Like this"
]

DONT_KNOW_RESPONSES = [
    "Not sure about that one.",
    "I don't know about that.",
    "No clue on that one.",
    "Can't help you with that.",
    "Don't know about that.",
    "Beats me.",
    "No idea about that one."
]

# Ollama Generation Parameters
DEFAULT_GENERATION_PARAMS = {
    'temperature': 0.8,
    'top_k': 40,
    'top_p': 0.9,
    'repeat_penalty': 1.3,
    'repeat_last_n': 64,
    'presence_penalty': 0.5,
    'frequency_penalty': 0.5,
    'stop': [
        '[Context]',
        '[Question]',
        '[Answer]',
        '\n\n[',
        '###',
        '[END]'
    ]
}

def get_web_context(query: str) -> str:
    """Get relevant web context for queries"""
    try:
        print("\n=== Search Request ===")
        print(f"Original query: {query}")
        
        # Clean the query
        clean_query = query.strip()
        print(f"Cleaned query: {clean_query}")
        
        # Prepare Serper API request
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': clean_query,
            'num': 3
        }
        
        response = requests.post(SERPER_API, json=payload, headers=headers)
        response.raise_for_status()
        
        results = response.json()
        print(f"\n=== Serper Response ===\nRaw response: {json.dumps(results, indent=2)[:500]}...")
        
        # Process results in order of priority
        if 'answerBox' in results:
            return process_answer_box(results['answerBox'])
        
        if 'knowledgeGraph' in results:
            return process_knowledge_graph(results['knowledgeGraph'])
            
        if 'organic' in results and results['organic']:
            return process_organic_results(results['organic'])
        
        print("No relevant results found")
        return "[NO_RESULTS] No current information available from search."
        
    except Exception as e:
        print(f"\n=== Search Error ===\nError: {str(e)}")
        return "Error: Unable to fetch search results."

def process_answer_box(answer: Dict[str, Any]) -> str:
    """Process answer box results from Serper API"""
    if 'answer' in answer:
        title = answer.get('title', '').lower()
        if title == '1 bitcoin =':
            return f"[PRICE] {answer['answer']}"
        elif any(word in title for word in ['price', 'stock', 'market', 'value']):
            return f"[PRICE] {answer['title']}: {answer['answer']}"
        elif 'weather' in title:
            return f"[WEATHER] {answer['answer']}"
        else:
            return f"[DIRECT_ANSWER] {answer['answer']}"
    elif 'snippet' in answer:
        return f"[SEARCH_RESULT] {answer['snippet']}"
    return ""

def process_knowledge_graph(kg: Dict[str, Any]) -> str:
    """Process knowledge graph results from Serper API"""
    kg_info = []
    for field in ['title', 'price', 'description', 'lastUpdated']:
        if field in kg:
            prefix = 'Current value: ' if field == 'price' else 'Last updated: ' if field == 'lastUpdated' else ''
            kg_info.append(f"{prefix}{kg[field]}")
    
    return f"[KNOWLEDGE_GRAPH] {' - '.join(kg_info)}" if kg_info else ""

def process_organic_results(organic: List[Dict[str, Any]]) -> str:
    """Process organic search results from Serper API"""
    if organic and 'snippet' in organic[0]:
        return f"[ORGANIC_RESULT] {organic[0]['snippet']}"
    return ""

def clean_response(response_text: str) -> str:
    """Clean and format the response text"""
    if not response_text:
        print("DEBUG: Empty response text")
        return random.choice(DONT_KNOW_RESPONSES)
        
    # Remove quotes if they wrap the entire response
    response_text = response_text.strip('"')
    
    # Remove system prompt if it appears anywhere in the response
    system_markers = [
        'PERSONALITY:',
        'RESPONSE STYLE:',
        'CONTEXT HANDLING:',
        'FORBIDDEN:',
        'You are Pearl'
    ]
    
    for marker in system_markers:
        if marker in response_text:
            # Take everything before the first system marker
            response_text = response_text.split(marker)[0].strip()
    
    # Remove format markers
    for marker in FORMAT_MARKERS:
        if response_text.startswith(marker):
            print(f"DEBUG: Removed format marker: {marker}")
            response_text = response_text.replace(marker, '').strip()
    
    # Remove self-questioning patterns
    response_text = re.sub(r'\n\n\[Question\].*?\[Answer\]', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\[Question\].*?\[Answer\]', '', response_text, flags=re.DOTALL)
    
    # Remove any remaining markdown-style tags
    response_text = re.sub(r'\[(Context|Question|Answer|Response)\]', '', response_text)
    
    # Split into paragraphs and clean each one
    paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
    if not paragraphs:
        print("DEBUG: No paragraphs after cleaning")
        return random.choice(DONT_KNOW_RESPONSES)
        
    # Rejoin paragraphs that are part of a list or instructions
    cleaned_paragraphs = []
    current_paragraph = []
    
    for p in paragraphs:
        # If it's a continuation of a list or steps, append to current
        if (p.strip().startswith(('-', '*', '•')) or 
            p.strip()[0].isdigit() or 
            current_paragraph and current_paragraph[-1].endswith(':')):
            current_paragraph.append(p)
        else:
            # If we have a current paragraph, save it
            if current_paragraph:
                cleaned_paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
            current_paragraph.append(p)
    
    # Add any remaining paragraph
    if current_paragraph:
        cleaned_paragraphs.append('\n'.join(current_paragraph))
    
    # Join all paragraphs
    response_text = '\n\n'.join(cleaned_paragraphs)
    
    # Handle empty responses
    if not response_text.strip():
        print("DEBUG: Empty response after cleaning")
        return random.choice(DONT_KNOW_RESPONSES)
    
    # Don't add periods to responses that are already properly terminated
    if not any(response_text.endswith(x) for x in ['.', '!', '?', ':', '-']):
        response_text = response_text + '.'
    
    return response_text.strip()

@app.route('/generate', methods=['POST'])
def generate():
    """Handle generation requests"""
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

        original_prompt = data['prompt']
        
        # Handle search queries
        should_search = (
            original_prompt.lower().startswith(SEARCH_PREFIX) or
            any(word in original_prompt.lower() for word in PRICE_RELATED_TERMS)
        )
        
        web_context = ""
        if should_search:
            clean_query = original_prompt[7:].strip() if original_prompt.lower().startswith(SEARCH_PREFIX) else original_prompt
            web_context = get_web_context(clean_query)
        
        cleaned_prompt = original_prompt[7:].strip() if original_prompt.lower().startswith(SEARCH_PREFIX) else original_prompt

        # Add user message to conversation
        conversation.add_message("user", cleaned_prompt)
        
        # Add web context if available
        if web_context:
            match = re.search(r'\[(PRICE|WEATHER|DIRECT_ANSWER|SEARCH_RESULT|KNOWLEDGE_GRAPH|ORGANIC_RESULT)\]\s*(.+)', web_context)
            web_context_content = match.group(2).strip() if match else web_context.strip()
            system_prompt = f"You must use this accurate web search result to answer the user's question: {web_context_content}"
        else:
            system_prompt = None

        # Get formatted context from previous messages
        context_text = conversation.get_formatted_context()

        # Prepare Ollama request with context
        ollama_request = {
            'model': MODEL_NAME,
            'prompt': cleaned_prompt,
            'context': conversation.context_tokens,
            'system': system_prompt,
            'stream': False,
            **DEFAULT_GENERATION_PARAMS
        }

        # Get response from Ollama
        response = requests.post(OLLAMA_API, json=ollama_request)
        response.raise_for_status()

        result = response.json()
        print(f"\n=== Raw Ollama Response ===\nResponse: {result.get('response', '')}")
        response_text = clean_response(result.get('response', '').strip())
        
        # Update context tokens from response
        conversation.context_tokens = result.get('context')
        
        # Store assistant response in conversation
        conversation.add_message("assistant", response_text)

        response_data = {
            'response': response_text,
            'model': MODEL_NAME,
            'context_length': len(conversation.messages),
            'has_web_context': bool(web_context)
        }

        # Log response
        print("\n=== Outgoing Response ===")
        print(f"Status Code: 200")
        print(f"Response Data: {response_data}")
        print(f"Web Context Used: {bool(web_context)}")
        print(f"Current Context Length: {len(conversation.messages)}")
        print("=====================\n")

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        error_response = {
            'error': f"Failed to communicate with Ollama",
            'details': str(e)
        }
        print(f"\n=== API Error ===\nError: {str(e)}")
        return jsonify(error_response), 503

    except Exception as e:
        error_response = {
            'error': 'Internal server error',
            'details': str(e)
        }
        print(f"\n=== Server Error ===\nError: {str(e)}")
        return jsonify(error_response), 500

if __name__ == '__main__':
    print(f"Starting server with model: {MODEL_NAME}")
    app.run(host='0.0.0.0', port=5000)