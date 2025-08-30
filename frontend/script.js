class ChatInterface {
    constructor() {
        // Sarvam API Configuration
        this.SARVAM_API_KEY = 'sk_ni27i31u_PTwALTVGFh9QFsPr2tNn9Eth';
        this.SARVAM_API_URL = 'https://api.sarvam.ai/translate';
        // Backend for RAG chat
        this.NEW_RAG_BASE = 'http://localhost:5050/api/newrag';
        this.init();
    }

    init() {
        this.setupTheme();
        this.setupEventListeners();
        this.setupMessageActions();
    }

    setupTheme() {
        // Check for saved theme preference or default to light mode
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        
        // Theme toggle functionality
        const themeToggle = document.getElementById('themeToggle');
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }

    setupEventListeners() {
        // Menu dropdown
        const menuButton = document.getElementById('menuButton');
        const dropdownMenu = document.getElementById('dropdownMenu');
        
        menuButton.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdownMenu.classList.toggle('show');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            dropdownMenu.classList.remove('show');
        });

        // Menu actions
        document.getElementById('clearChat').addEventListener('click', () => {
            this.clearChat();
            dropdownMenu.classList.remove('show');
        });

        document.getElementById('exportChat').addEventListener('click', () => {
            this.exportChat();
            dropdownMenu.classList.remove('show');
        });

        // Send message
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');

        sendButton.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    setupMessageActions() {
        // Setup hover dropdowns for existing messages
        this.setupMessageDropdowns();
    }

    setupMessageDropdowns() {
        const messages = document.querySelectorAll('.ai-message');
        
        messages.forEach(message => {
            const actionButton = message.querySelector('.action-button');
            const dropdown = message.querySelector('.message-dropdown');
            
            if (actionButton && dropdown) {
                actionButton.addEventListener('click', (e) => {
                    e.stopPropagation();
                    // Close other dropdowns
                    document.querySelectorAll('.message-dropdown.show').forEach(d => {
                        if (d !== dropdown) d.classList.remove('show');
                    });
                    dropdown.classList.toggle('show');
                });

                // Setup translate main button and submenu
                const translateMainBtn = dropdown.querySelector('.translate-main-btn');
                const languageSubmenu = dropdown.querySelector('.language-submenu');
                const translateBtns = dropdown.querySelectorAll('.translate-btn');
                const speakBtn = dropdown.querySelector('.speak-btn');

                if (translateMainBtn && languageSubmenu) {
                    translateMainBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        languageSubmenu.classList.toggle('show');
                    });
                }

                translateBtns.forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const targetLang = e.target.closest('.translate-btn').dataset.lang;
                        this.translateMessage(message, targetLang);
                        dropdown.classList.remove('show');
                        if (languageSubmenu) {
                            languageSubmenu.classList.remove('show');
                        }
                    });
                });

                if (speakBtn) {
                    speakBtn.addEventListener('click', () => {
                        this.speakMessage(message);
                        dropdown.classList.remove('show');
                    });
                }
            }
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            document.querySelectorAll('.message-dropdown.show').forEach(dropdown => {
                dropdown.classList.remove('show');
            });
            document.querySelectorAll('.language-submenu.show').forEach(submenu => {
                submenu.classList.remove('show');
            });
        });
    }

    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        messageInput.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        // Prepare an empty AI message to stream into
        const aiMsgEl = this.addMessage('', 'ai');
        const aiTextEl = aiMsgEl.querySelector('.message-text');

        // Stream from backend
        (async () => {
            try {
                const res = await fetch(`${this.NEW_RAG_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message }),
                });
                if (!res.ok || !res.body) {
                    throw new Error(`Chat API error: ${res.status}`);
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                this.hideTypingIndicator();
                let done = false;
                while (!done) {
                    const { value, done: doneChunk } = await reader.read();
                    done = doneChunk;
                    if (value) {
                        const chunk = decoder.decode(value, { stream: !done });
                        aiTextEl.textContent += chunk;
                        this.scrollToBottom();
                    }
                }
                // Ensure dropdowns are wired for the new AI message
                this.setupMessageDropdowns();
            } catch (err) {
                console.error('Chat stream failed:', err);
                this.hideTypingIndicator();
                aiTextEl.textContent = 'Sorry, I had trouble connecting to the assistant.';
            }
        })();
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    addMessage(text, type) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        if (type === 'ai') {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="message-text">${text}</div>
                    <div class="message-actions">
                        <button class="action-button" title="More options">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6,9 12,15 18,9"></polyline>
                            </svg>
                        </button>
                        <div class="message-dropdown">
                            <button class="dropdown-item speak-btn">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                    <path d="m19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                                </svg>
                                Speak
                            </button>
                            <button class="dropdown-item translate-main-btn">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="m5 8 6 6"></path>
                                    <path d="m4 14 6-6 2-3"></path>
                                    <path d="M2 5h12"></path>
                                    <path d="M7 2h1"></path>
                                    <path d="m22 22-5-10-5 10"></path>
                                    <path d="M14 18h6"></path>
                                </svg>
                                Translate
                                <svg class="arrow-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="9,18 15,12 9,6"></polyline>
                                </svg>
                            </button>
                            
                            <!-- Language Submenu -->
                            <div class="language-submenu">
                                <button class="dropdown-item translate-btn" data-lang="hi">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="m5 8 6 6"></path>
                                        <path d="m4 14 6-6 2-3"></path>
                                        <path d="M2 5h12"></path>
                                        <path d="M7 2h1"></path>
                                        <path d="m22 22-5-10-5 10"></path>
                                        <path d="M14 18h6"></path>
                                    </svg>
                                    हिंदी
                                </button>
                                <button class="dropdown-item translate-btn" data-lang="ta">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="m5 8 6 6"></path>
                                        <path d="m4 14 6-6 2-3"></path>
                                        <path d="M2 5h12"></path>
                                        <path d="M7 2h1"></path>
                                        <path d="m22 22-5-10-5 10"></path>
                                        <path d="M14 18h6"></path>
                                    </svg>
                                    தமிழ்
                                </button>
                                <button class="dropdown-item translate-btn" data-lang="te">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="m5 8 6 6"></path>
                                        <path d="m4 14 6-6 2-3"></path>
                                        <path d="M2 5h12"></path>
                                        <path d="M7 2h1"></path>
                                        <path d="m22 22-5-10-5 10"></path>
                                        <path d="M14 18h6"></path>
                                    </svg>
                                    తెలుగు
                                </button>
                                <button class="dropdown-item translate-btn" data-lang="en">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="m5 8 6 6"></path>
                                        <path d="m4 14 6-6 2-3"></path>
                                        <path d="M2 5h12"></path>
                                        <path d="M7 2h1"></path>
                                        <path d="m22 22-5-10-5 10"></path>
                                        <path d="M14 18h6"></path>
                                    </svg>
                                    English
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="message-text">${text}</div>
                </div>
            `;
        }

        chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // Setup dropdown for new AI message
        if (type === 'ai') {
            this.setupMessageDropdowns();
        }

        return messageDiv;
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';
        console.log('Chat cleared');
    }

    exportChat() {
        const messages = document.querySelectorAll('.message');
        let chatContent = 'Chat Export\n==========\n\n';
        
        messages.forEach(message => {
            const type = message.classList.contains('ai-message') ? 'AI' : 'User';
            const text = message.querySelector('.message-text').textContent;
            chatContent += `${type}: ${text}\n\n`;
        });

        const blob = new Blob([chatContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-export-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Chat exported');
    }

    async detectLanguage(text) {
        // Simple language detection based on script/characters
        const hindiRegex = /[\u0900-\u097F]/;
        const tamilRegex = /[\u0B80-\u0BFF]/;
        const teluguRegex = /[\u0C00-\u0C7F]/;
        
        if (hindiRegex.test(text)) return 'hi-IN';
        if (tamilRegex.test(text)) return 'ta-IN';
        if (teluguRegex.test(text)) return 'te-IN';
        return 'en-IN'; // Default to English
    }

    async translateWithSarvam(text, targetLang, sourceLang = null) {
        try {
            // Auto-detect source language if not provided
            if (!sourceLang) {
                sourceLang = await this.detectLanguage(text);
            }

            // Skip translation if source and target are the same
            if (sourceLang === targetLang) {
                return text;
            }

            const response = await fetch(this.SARVAM_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'API-Subscription-Key': this.SARVAM_API_KEY
                },
                body: JSON.stringify({
                    input: text,
                    source_language_code: sourceLang,
                    target_language_code: targetLang,
                    speaker_gender: "Male",
                    mode: "formal",
                    model: "mayura:v1",
                    enable_preprocessing: true
                })
            });

            if (!response.ok) {
                throw new Error(`Translation API error: ${response.status}`);
            }

            const data = await response.json();
            return data.translated_text || text;
        } catch (error) {
            console.error('Translation error:', error);
            // Fallback to original text if translation fails
            return `[Translation Error] ${text}`;
        }
    }

    async translateMessage(messageElement, targetLang) {
        const messageText = messageElement.querySelector('.message-text');
        const originalText = messageText.textContent;
        
        // Show loading indicator
        const originalContent = messageText.textContent;
        messageText.textContent = 'Translating...';
        
        try {
            // Map UI language codes to Sarvam API codes
            const langMap = {
                'hi': 'hi-IN',
                'ta': 'ta-IN', 
                'te': 'te-IN',
                'en': 'en-IN'
            };
            
            const targetApiLang = langMap[targetLang];
            const translatedText = await this.translateWithSarvam(originalText, targetApiLang);
            
            messageText.textContent = translatedText;
            
            // Store original text for potential re-translation
            if (!messageElement.dataset.originalText) {
                messageElement.dataset.originalText = originalContent;
            }
            
            console.log(`Message translated to ${targetLang}:`, originalText, '->', translatedText);
        } catch (error) {
            console.error('Translation failed:', error);
            messageText.textContent = originalContent; // Restore original text
            alert('Translation failed. Please try again.');
        }
    }

    speakMessage(messageElement) {
        const messageText = messageElement.querySelector('.message-text').textContent;
        
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            speechSynthesis.cancel();
            
            // Show bot avatar
            this.showBotAvatar();
            
            const utterance = new SpeechSynthesisUtterance(messageText);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            utterance.volume = 1;
            
            // Start speaking animation
            utterance.onstart = () => {
                this.startSpeakingAnimation();
            };
            
            // Stop speaking animation when done
            utterance.onend = () => {
                this.stopSpeakingAnimation();
                setTimeout(() => {
                    this.hideBotAvatar();
                }, 1000);
            };
            
            // Handle errors
            utterance.onerror = () => {
                this.stopSpeakingAnimation();
                this.hideBotAvatar();
            };
            
            speechSynthesis.speak(utterance);
            console.log('Speaking message:', messageText);
        } else {
            console.log('Speech synthesis not supported in this browser');
            alert('Speech synthesis is not supported in your browser');
        }
    }

    showBotAvatar() {
        const botAvatar = document.getElementById('botAvatar');
        botAvatar.classList.add('show');
    }

    hideBotAvatar() {
        const botAvatar = document.getElementById('botAvatar');
        botAvatar.classList.remove('show', 'speaking');
    }

    startSpeakingAnimation() {
        const botAvatar = document.getElementById('botAvatar');
        botAvatar.classList.add('speaking');
    }

    stopSpeakingAnimation() {
        const botAvatar = document.getElementById('botAvatar');
        botAvatar.classList.remove('speaking');
    }
}

// Initialize the chat interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
});