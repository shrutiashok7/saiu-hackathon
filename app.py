from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv
import chromadb
import requests
import json
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='frontend', static_url_path='')

# --- Configuration ---
OLLAMA_CHAT_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model choices
GENERATION_MODEL = "mistral:latest"  # Ollama model (still used for embeddings-assisted routes if needed)
EMBEDDING_MODEL = "nomic-embed-text:latest"
HELPER_MODEL = "mistral:latest"
GUIDANCE_SEARCH_MODEL = "llama-3-sonar-large-32k-online"  # Perplexity web-enabled model
PRIMARY_CHAT_MODEL = "llama-3.1-sonar-small-128k-online"  # Perplexity primary chat model
OPENROUTER_FALLBACK_MODEL = "openai/gpt-oss-20b:free"

# Construct the absolute path to the chroma_db directory relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "pdf_embeddings"

# --- Chatbot Logic ---
class Chatbot:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            print("Successfully connected to ChromaDB collection.")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            self.collection = None
        self.chat_history = []
        self.user_profile = {"major": None, "ambition": None}

    def clear_session(self):
        self.chat_history = []
        self.user_profile = {"major": None, "ambition": None}
        print("Chat session cleared.")

    def process_query(self, query):
        chat_history_str = self._format_chat_history(self.chat_history)
        routing_result = self._route_and_refine_query(query, chat_history_str)
        intent = routing_result["intent"]

        if intent == 'guidance_search':
            if not self.user_profile.get('ambition'):
                # This part needs to be handled by the frontend logic now
                return "That's an important question. To give you the best guidance, could you tell me a bit about your career ambitions or what you hope to achieve?"
            else:
                return self._get_news_guidance(query, self.user_profile)

        elif intent == 'retrieval':
            context = self._retrieve_context(routing_result["query"], self.collection)
            self.retrieval_prompt = """
You are an expert academic counselor. Your task is to answer student questions based on the provided course catalog context. 
The context contains information about courses, prerequisites, credit hours, and instructors.
When a student asks a question, you must find the answer within the provided context and present it clearly.
If the answer is not in the context, state that you don't have enough information and suggest they contact a human advisor.
Format your response using Markdown for readability (headings, bold, lists, etc.).
IMPORTANT: Do not include any links in your response. Provide text-only answers.
"""
            user_prompt_with_context = f'Context from university documents:\n---\n{context or "No context was found for this query."}\n---\nBased on the context above, please answer my last question: "{query}" '
            messages_for_api = [{"role": "system", "content": self.retrieval_prompt}] + self.chat_history + [{"role": "user", "content": user_prompt_with_context}]
            return self._stream_perplexity_or_openrouter(messages_for_api)

        else: # 'conversation'
            persona_prefix = {"role": "system", "content": "You are Nexus, a friendly and helpful AI academic counsellor for Sai University. Please use simple Markdown to format your responses where appropriate (e.g., lists, bold text). IMPORTANT: Do not include any links in your response. Provide text-only answers."}
            messages_for_api = [persona_prefix] + self.chat_history + [{"role": "user", "content": query}]
            return self._stream_ollama_chat_response(GENERATION_MODEL, messages_for_api)

    def _get_ollama_embedding(self, prompt):
        try:
            payload = {"model": EMBEDDING_MODEL, "prompt": prompt}
            response = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload)
            response.raise_for_status()
            return response.json().get("embedding")
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding from Ollama: {e}")
            return None

    def _stream_perplexity_or_openrouter(self, messages):
        """Try streaming from Perplexity first; on failure, fall back to OpenRouter."""
        def generate():
            # 1) Try Perplexity
            if PERPLEXITY_API_KEY:
                headers = {
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": PRIMARY_CHAT_MODEL,
                    "messages": messages,
                    "stream": True,
                }
                try:
                    with requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            s = line.decode('utf-8', errors='ignore')
                            if not s.startswith('data: '):
                                continue
                            try:
                                data = json.loads(s[6:])
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except Exception:
                                continue
                        return
                except Exception as e:
                    print(f"Perplexity stream failed, falling back to OpenRouter: {e}")

            # 2) Fallback to OpenRouter
            if not OPENROUTER_API_KEY:
                yield "Sorry, no AI provider is configured."
                return
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": OPENROUTER_FALLBACK_MODEL,
                "messages": messages,
                "stream": True,
            }
            try:
                with requests.post(OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        s = line.decode('utf-8', errors='ignore')
                        if not s.startswith('data: '):
                            continue
                        try:
                            data = json.loads(s[6:])
                            delta = data.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                        except Exception:
                            continue
            except Exception as e:
                print(f"OpenRouter stream failed: {e}")
                yield "Sorry, I encountered a connection error."

        return Response(generate(), mimetype='text/plain')

    def _route_and_refine_query(self, query, chat_history_str):
        system_prompt = """You are an expert query analysis agent...""" # Truncated for brevity
        user_prompt = f'Conversation History:\n{chat_history_str}\n\nUser Query: "{query}"\n\nYour JSON Output:'
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            payload = {"model": HELPER_MODEL, "messages": messages, "stream": False, "format": "json"}
            response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload)
            response.raise_for_status()
            response_data = response.json()
            json_content = json.loads(response_data.get('message', {}).get('content', '{}'))
            return {"intent": json_content.get("intent", "conversation"), "query": json_content.get("query")}
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Error in routing/refining query: {e}")
            return {"intent": "conversation", "query": query}

    def _get_news_guidance(self, query, user_profile):
        if not PERPLEXITY_API_KEY:
            return Response("The web search feature is not configured.", mimetype='text/plain')
        headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
        system_prompt = "You are a helpful AI career and academic assistant. Format your response in Markdown, using headings and lists to make it easy to read. IMPORTANT: Do not include any links in your response. Provide text-only answers."
        user_context = f"- My Major: {user_profile.get('major', 'Not specified')}\n- My Ambition: {user_profile.get('ambition', 'Not specified')}\n\nMy Question: {query}"
        payload = {"model": GUIDANCE_SEARCH_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context}], "stream": True}
        def generate():
            try:
                with requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, stream=True) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line and line.decode('utf-8').startswith('data: '):
                            try:
                                json_data = json.loads(line.decode('utf-8')[6:])
                                content = json_data['choices'][0]['delta'].get('content', '')
                                if content:
                                    yield content
                            except (json.JSONDecodeError, KeyError):
                                continue
            except requests.exceptions.RequestException as e:
                print(f"Error during web search: {e}")
                yield "Sorry, I couldn't perform the web search."
        return Response(generate(), mimetype='text/plain')

    def _retrieve_context(self, query, collection):
        if not query or not collection: return ""
        query_embedding = self._get_ollama_embedding(query)
        if not query_embedding: return ""
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        if not results or not results.get('documents'): return ""
        return "\n\n---\n\n".join(results['documents'][0])

    def _format_chat_history(self, history):
        if not history: return "No previous conversation."
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

chatbot = Chatbot()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Here we get a generator/Response object
    response_stream = chatbot.process_query(user_message)
    
    # We need to handle the streaming response properly
    return response_stream

@app.route('/clear', methods=['POST'])
def clear():
    chatbot.clear_session()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
