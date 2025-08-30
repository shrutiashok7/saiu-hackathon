import os
import json
import requests
import chromadb
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv

"""
New RAG Backend (derived from newragsearch.py)

- Streams chat responses via Flask Response generators
- Routing:
  - guidance_search -> Perplexity streaming (web search model)
  - retrieval -> Ollama generation with retrieved context from ChromaDB
  - conversation -> Ollama generation
- Embeddings via Ollama

Endpoints:
- POST /api/newrag/chat  {"message": string}
  Streams text/plain chunks
- POST /api/newrag/clear  clears in-memory session
- GET  /api/newrag/health health check

Set environment variables:
- PERPLEXITY_API_KEY
- (Ollama must be running locally)
"""

load_dotenv()

app = Flask(__name__)

# Allow simple CORS for local development (frontend served from a different port)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# --- Configuration ---
# Ollama API endpoints
OLLAMA_CHAT_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"

# Perplexity API
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
 
# OpenRouter (fallback)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models
GENERATION_MODEL = "mistral:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
HELPER_MODEL = "mistral:latest"
GUIDANCE_SEARCH_MODEL = "llama-3-sonar-large-32k-online"
OPENROUTER_FALLBACK_MODEL = "openai/gpt-oss-20b:free"

# ChromaDB configuration (absolute path relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "pdf_embeddings"


class NewRAGService:
    def __init__(self):
        # Connect to ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            print("[NewRAG] Connected to ChromaDB collection.")
        except Exception as e:
            print(f"[NewRAG] Error connecting to ChromaDB: {e}")
            self.collection = None
        # Simple in-memory session
        self.chat_history = []
        self.user_profile = {"major": None, "ambition": None}
        self.awaiting_ambition = False

    # -------- Embeddings / Retrieval --------
    def _get_ollama_embedding(self, prompt: str):
        try:
            payload = {"model": EMBEDDING_MODEL, "prompt": prompt}
            resp = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("embedding")
        except requests.exceptions.RequestException as e:
            print(f"[NewRAG] Embedding error: {e}")
            return None

    def _retrieve_context(self, query: str):
        if not query or not self.collection:
            return ""
        embedding = self._get_ollama_embedding(query)
        if not embedding:
            return ""
        results = self.collection.query(query_embeddings=[embedding], n_results=5)
        if not results or not results.get("documents"):
            return ""
        return "\n\n---\n\n".join(results["documents"][0])

    # -------- LLM Streaming --------
    def _stream_ollama(self, messages):
        def generate():
            try:
                payload = {"model": GENERATION_MODEL, "messages": messages, "stream": True}
                with requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
            except requests.exceptions.RequestException as e:
                print(f"[NewRAG] Ollama stream error: {e}")
                yield "Sorry, I encountered a connection error."
        return Response(generate(), mimetype="text/plain")

    def _stream_perplexity_or_openrouter(self, system_prompt: str, user_content: str):
        """Try Perplexity first; if it fails or not configured, fall back to OpenRouter."""
        def generate():
            # 1) Perplexity attempt
            if PERPLEXITY_API_KEY:
                headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": GUIDANCE_SEARCH_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": True,
                }
                try:
                    with requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            s = line.decode("utf-8", errors="ignore")
                            if not s.startswith("data: "):
                                continue
                            try:
                                data = json.loads(s[6:])
                                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue
                        return
                except Exception as e:
                    print(f"[NewRAG] Perplexity stream error, falling back to OpenRouter: {e}")

            # 2) OpenRouter fallback
            if not OPENROUTER_API_KEY:
                yield "The web search feature is not configured."
                return
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": OPENROUTER_FALLBACK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": True,
            }
            try:
                with requests.post(OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        s = line.decode("utf-8", errors="ignore")
                        if not s.startswith("data: "):
                            continue
                        try:
                            data = json.loads(s[6:])
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
            except Exception as e:
                print(f"[NewRAG] OpenRouter stream error: {e}")
                yield "Sorry, I couldn't perform the web search."
        return Response(generate(), mimetype="text/plain")

    # -------- Router --------
    def _format_history(self):
        if not self.chat_history:
            return "No previous conversation."
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history])

    def _route_and_refine(self, query: str):
        system_prompt = (
            "You are an expert query analysis agent. Your task is to analyze a user's query and conversation history, "
            "then output a JSON object with two fields: 'intent' and 'query'.\n\n"
            "1. 'intent': retrieval | guidance_search | conversation\n"
            "2. 'query': If 'retrieval', rewrite the query for vector search; otherwise null.\n\n"
            "IMPORTANT: Output must be a single valid JSON object."
        )
        user_prompt = f"Conversation History:\n{self._format_history()}\n\nUser Query: \"{query}\"\n\nYour JSON Output:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            payload = {"model": HELPER_MODEL, "messages": messages, "stream": False, "format": "json"}
            resp = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            json_content = json.loads(data.get("message", {}).get("content", "{}"))
            intent = json_content.get("intent", "conversation")
            refined = json_content.get("query")
            return {"intent": intent, "query": refined}
        except Exception as e:
            print(f"[NewRAG] Router error: {e}")
            return {"intent": "conversation", "query": query}

    # -------- Public handlers --------
    def handle_message(self, user_text: str):
        # Handle ambition follow-up
        if self.awaiting_ambition:
            self.user_profile["ambition"] = user_text
            self.awaiting_ambition = False
            reply = (
                "Thank you for sharing. I've made a note of that. Now, what was your original question about your future?"
            )
            self._append_history(user_text, reply)
            def gen_once():
                yield reply
            return Response(gen_once(), mimetype="text/plain")

        # Route
        routing = self._route_and_refine(user_text)
        intent = routing["intent"]

        if intent == "guidance_search":
            if not self.user_profile.get("ambition"):
                self.awaiting_ambition = True
                reply = (
                    "That's an important question. To give you the best guidance, could you tell me a bit about your career ambitions or what you hope to achieve?"
                )
                self._append_history(user_text, reply)
                def gen_once():
                    yield reply
                return Response(gen_once(), mimetype="text/plain")
            # Use Perplexity
            system_prompt = (
                "You are a helpful AI career and academic assistant. Provide a concise, web-informed summary. "
                "Format with simple Markdown (headings, lists). Do not include links."
            )
            user_ctx = (
                f"- My Major: {self.user_profile.get('major', 'Not specified')}\n"
                f"- My Ambition: {self.user_profile.get('ambition', 'Not specified')}\n\n"
                f"My Question: {user_text}"
            )
            # Note: history is not embedded into Perplexity call beyond this for simplicity
            return self._stream_perplexity_or_openrouter(system_prompt, user_ctx)

        if intent == "retrieval":
            context = self._retrieve_context(routing.get("query"))
            system_prompt = (
                "You are 'Nexus,' the AI Academic Counsellor for Sai University. Use ONLY the provided context. "
                "Format in clear Markdown with headings and bullet points. Do not include links."
            )
            user_with_ctx = (
                f"Context from university documents:\n---\n{context or 'No context was found for this query.'}\n---\n"
                f"Based on the context above and our prior conversation, please answer my last question: \"{user_text}\" "
            )
            messages = [{"role": "system", "content": system_prompt}] + self.chat_history + [{"role": "user", "content": user_with_ctx}]
            return self._stream_ollama(messages)

        # conversation
        persona = {
            "role": "system",
            "content": "You are Nexus, a friendly and helpful AI academic counsellor for Sai University. Do not include links.",
        }
        messages = [persona] + self.chat_history + [{"role": "user", "content": user_text}]
        return self._stream_ollama(messages)

    def _append_history(self, user_text: str, assistant_text: str):
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": assistant_text})

    def clear(self):
        self.chat_history = []
        self.user_profile = {"major": None, "ambition": None}
        self.awaiting_ambition = False


service = NewRAGService()


# -------- Flask Endpoints --------
@app.route("/", methods=["GET"])
def index():
    return (
        "New RAG Backend is running. See /api/newrag/health for status.",
        200,
        {"Content-Type": "text/plain"},
    )

@app.route("/api/newrag/health", methods=["GET"])
def health():
    status = {
        "chroma_connected": bool(service.collection),
        "perplexity_configured": bool(PERPLEXITY_API_KEY),
    }
    return jsonify(status)


@app.route("/api/newrag/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()
    # Optional: update profile if provided
    profile = data.get("profile")
    if isinstance(profile, dict):
        service.user_profile.update({k: v for k, v in profile.items() if k in ("major", "ambition")})
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    return service.handle_message(msg)


@app.route("/api/newrag/clear", methods=["POST"])
def clear():
    service.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
