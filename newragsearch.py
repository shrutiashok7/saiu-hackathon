import chromadb
import requests
import json
import os

# --- Configuration ---
# Ollama API endpoints
OLLAMA_CHAT_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"

# Perplexity API - add your key here or set as an environment variable
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "") 
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


# Models
GENERATION_MODEL = "mistral:latest" 
EMBEDDING_MODEL = "nomic-embed-text:latest" 
HELPER_MODEL = "mistral:latest"
# Perplexity model for web search
GUIDANCE_SEARCH_MODEL = "llama-3-sonar-large-32k-online"

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_embeddings"


def get_ollama_embedding(prompt):
    """Gets an embedding from Ollama."""
    try:
        payload = {"model": EMBEDDING_MODEL, "prompt": prompt}
        response = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json().get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"\nError getting embedding from Ollama: {e}")
        return None

def stream_ollama_chat_response(model, messages):
    """Streams a chat response from Ollama, yielding content chunks."""
    try:
        payload = {"model": model, "messages": messages, "stream": True}
        with requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        print(f"\nError streaming from Ollama API: {e}")
        yield "Sorry, I encountered a connection error."

def route_and_refine_query(query, chat_history_str):
    """
    Analyzes the user's query and decides the best course of action.
    """
    system_prompt = """You are an expert query analysis agent. Your task is to analyze a user's query and conversation history, then output a JSON object with two fields: "intent" and "query".

1.  *"intent"*: Classify the user's primary goal into one of three categories:
    * **retrieval**: If the query asks for specific, factual information that would likely be found in a local document (like a university course catalog, e.g., "details on CSE-412", "prerequisites for AI course").
    * **guidance_search**: If the query is broad, open-ended, or future-focused, requiring up-to-date, external information. This includes questions about career paths, future skills, market trends, or "what is best for me".
    * **conversation**: If the query is a general greeting, a social question, or part of a casual chat that doesn't fit the other categories.

2.  *"query"*:
    * If the intent is retrieval, rewrite the query to be an optimized search term for a vector database.
    * If the intent is guidance_search or conversation, set this field to null.

IMPORTANT: Your output MUST be a single, valid JSON object and nothing else.
"""
    user_prompt = f"""Conversation History:
{chat_history_str}

User Query: "{query}"

Your JSON Output:"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        payload = {"model": HELPER_MODEL, "messages": messages, "stream": False, "format": "json"}
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        json_content = json.loads(response_data.get('message', {}).get('content', '{}'))
        
        intent = json_content.get("intent", "conversation")
        refined_query = json_content.get("query")
        
        print(f"  [Router decision: {intent}]")
        if intent == "retrieval":
             print(f"  [Refined Query for Search: '{refined_query}']")

        return {"intent": intent, "query": refined_query}
        
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"\nError in routing/refining query: {e}")
        return {"intent": "conversation", "query": query}

def get_news_guidance(query: str, user_profile: dict) -> str:
    """
    Calls Perplexity API to get web-based guidance.
    """
    if not PERPLEXITY_API_KEY:
        return "The web search feature is not configured. Please add a Perplexity API key."

    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    system_prompt = (
        "You are a helpful AI career and academic assistant. Your task is to provide a concise, web-informed summary to answer the user's question, "
        "taking their academic major and career ambitions into account to personalize the response."
    )
    
    user_context = (
        f"Here is some information about me:\n"
        f"- My Major: {user_profile.get('major', 'Not specified')}\n"
        f"- My Ambition: {user_profile.get('ambition', 'Not specified')}\n\n"
        f"My Question: {query}"
    )

    payload = {
        "model": GUIDANCE_SEARCH_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ],
        "stream": True
    }
    
    try:
        with requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            json_data = json.loads(line_str[6:])
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue
    except requests.exceptions.RequestException as e:
        print(f"\nError during web search: {e}")
        yield "Sorry, I couldn't perform the web search."


def retrieve_context(query, collection):
    """Retrieves relevant context from ChromaDB based on the query."""
    if not query: return ""
    query_embedding = get_ollama_embedding(query)
    if not query_embedding:
        print("  [Could not generate embedding for retrieval.]")
        return ""
    
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    if not results or not results.get('documents'): return ""
        
    context = "\n\n---\n\n".join(results['documents'][0])
    return context

def format_chat_history(history):
    """Formats chat history into a simple string."""
    if not history: return "No previous conversation."
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

def main():
    """Main function to run the RAG chatbot."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Successfully connected to ChromaDB collection.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}\nPlease ensure 'pdf_embedder.py' has been run.")
        return

    chat_history = []
    user_profile = {"major": None, "ambition": None}
    awaiting_ambition = False

    print("\n--- Nexus AI Academic Counsellor ---")
    print("Ask about courses, or seek guidance for your future. Type '/clear' or '/exit'.")
    
    while True:
        query = input("\nYou: ")
        if not query: continue
        if query.lower() == '/exit': break
        if query.lower() == '/clear':
            chat_history, user_profile = [], {"major": None, "ambition": None}
            awaiting_ambition = False
            print("\n[Chat history and user profile cleared.]")
            continue

        final_response_parts = []
        print(f"\nAssistant: ", end="", flush=True)

        if awaiting_ambition:
            user_profile['ambition'] = query
            awaiting_ambition = False
            response_text = "Thank you for sharing. I've made a note of that. Now, what was your original question about your future?"
            print(response_text)
            final_response_parts.append(response_text)
        else:
            chat_history_str = format_chat_history(chat_history)
            routing_result = route_and_refine_query(query, chat_history_str)
            intent = routing_result["intent"]

            if intent == 'guidance_search':
                if not user_profile.get('ambition'):
                    awaiting_ambition = True
                    response_text = "That's an important question. To give you the best guidance, could you tell me a bit about your career ambitions or what you hope to achieve?"
                    print(response_text)
                    final_response_parts.append(response_text)
                else:
                    print("  [Performing web search for guidance...]")
                    for chunk in get_news_guidance(query, user_profile):
                        print(chunk, end="", flush=True)
                        final_response_parts.append(chunk)

            elif intent == 'retrieval':
                print("  [Performing retrieval...]")
                context = retrieve_context(routing_result["query"], collection)
                system_prompt = """You are 'Nexus,' the dedicated AI Academic Counsellor for students at Sai University. Your core mission is to provide clear, accurate, and empowering guidance, helping students navigate their academic options and succeed in their studies. You act as a knowledgeable and friendly first point of contact.
*Your Identity & Persona:*
- *Name:* Nexus
- *Role:* AI Academic Counsellor at Sai University
- *Tone:* Professional, encouraging, patient, and highly supportive.
*Your Core Directives & Reasoning Process:*
Before you respond, always follow this internal protocol to ensure the highest quality of guidance:
1.  *Deconstruct the Student's Need:* What is the core of their question? Are they exploring new courses, checking prerequisites, confused about eligibility, or planning their semester?
2.  *Analyze the Official Context:* Meticulously review the provided university documents. Your answers MUST be grounded exclusively in this information. Identify all relevant details, even if they are in different sections of the context.
3.  *Synthesize and Structure for Clarity:* Present your findings in a structured, easy-to-digest format. Use bold headings (e.g., *Course Overview:, **Eligibility Requirements:, **Key Topics:*) and bullet points to make complex information simple.
4.  *Provide Definitive & Safe Eligibility Advice:* When asked about eligibility, quote the exact requirements from the documents. Conclude with a clear, safe summary like: "Based on the official requirements, you appear to be eligible," or "Based on the document, this course seems to require prerequisites you haven't mentioned. It's best to double-check." If the documents are unclear, state that directly.
5.  *Address Information Gaps Transparently:* If the answer is not in the provided documents, you MUST state this clearly. For example: "I can't find the specific assessment details in the documents I have access to. For that information, reaching out to the course instructor or the department office would be the best next step." Never invent or assume information.
6.  *Empower the Student:* Always conclude your response on a positive and helpful note. Ask if they have more questions or if there's another area you can assist with. Your goal is to make the student feel confident and well-informed.
"""
                user_prompt_with_context = f"""Context from university documents:\n---\n{context or "No context was found for this query."}\n---\nBased on the context above and our prior conversation, please answer my last question: "{query}" """
                messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_prompt_with_context}]
                for chunk in stream_ollama_chat_response(GENERATION_MODEL, messages_for_api):
                    print(chunk, end="", flush=True)
                    final_response_parts.append(chunk)

            else: # 'conversation'
                print("  [Generating conversational response...]")
                persona_prefix = {"role": "system", "content": "You are Nexus, a friendly and helpful AI academic counsellor for Sai University."}
                messages_for_api = [persona_prefix] + chat_history + [{"role": "user", "content": query}]
                for chunk in stream_ollama_chat_response(GENERATION_MODEL, messages_for_api):
                    print(chunk, end="", flush=True)
                    final_response_parts.append(chunk)

        print() 
        final_response = "".join(final_response_parts)

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()