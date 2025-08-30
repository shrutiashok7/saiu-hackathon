
# **Nexus – SaiU Conversational AI Chatbot**

Nexus is a conversational AI chatbot tailored for Sai University. It answers student queries about the SaiU course catalog, helps with academic planning & career guidance, and integrates advanced features like translation, summarization, and text-to-speech.
Built for the Hackathon – Track C (Advanced: Generative AI & LLMs).

## **Features:**
### Course Catalog Q&A
* Uses Retrieval-Augmented Generation (RAG) on the SaiU course catalog.
* Answers queries related to credits, courses, prerequisites, and academic information.
* Provides summarized, simplified explanations of course descriptions.

### Career Guidance
* Integrated with the Perplexity API to analyze latest industry trends & news.
* Suggests relevant career paths and course selections for skill-building.

### Multilingual Translation
* Supports English ↔ Tamil, Telugu, Hindi.
* Students can query in regional languages, and the bot translates seamlessly back to English for LLM processing.

### Text-to-Speech (TTS)
* Bot responses can be spoken aloud with one click.
* Fun UI: when “Speak” is pressed, a floating bot avatar appears to deliver the answer.

### Summarization
* Summarizes long course details into simple, easy-to-understand descriptions.
* Helps students grasp course requirements quickly.

## **Tech Stack:**
* LLM API: Perplexity API + OpenRouter (Google Gemini 2.5)
* RAG Database: ChromaDB
* Backend: Python
* Text-to-Speech: TTS API (integrated with bot interface)
* Translation: Language translation pipeline (English, Tamil, Telugu, Hindi)
* UI: Minimal, clean design (with animations, floating bot avatar)

## **Project Structure**
saiu-hackathon/
│── app/        # Core chatbot code (py)
│── data/       # SaiU course catalog (RAG source)  
│── ui/         # Chatbot UI (animations, floating bot) js, html, css
│── tests/      # Test scripts  
│── README.md   # Project documentation  

## **How It Works:**
1. User asks a question → “What are the CS electives available this semester?” or “How does this course help me in my biotech career?”
2. RAG Retrieval → ChromaDB searches SaiU catalog and retrieves relevant chunks.
3. LLM Processing → Query + context sent to Perplexity/OpenRouter model.
4. Enhanced Answer → Model returns course details + simplified explanation.
5. Optional Features:
    * Translate query/response (English ↔ Tamil/Telugu/Hindi).
    * Summarize long course descriptions.
    * Speak out response using TTS.
    * Floating bot avatar animation during speech.

### **Team:**
Built by 
1. https://github.com/lokeegit
2. https://github.com/Tiffanorsgit 
3. https://github.com/shrutiashok7
