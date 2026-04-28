# 💕 Truly Yours AI Chatbot

> **Your AI companion with web search capabilities** — powered by Groq, OpenRouter, and Brave Search API

## 🌟 What's New?

This fork adds **Brave Search API integration** with intelligent agent support, allowing compatible AI models to search the web autonomously when they need current information.

### ✨ Features

- 🤖 **Multiple AI Providers**: Choose between Groq (fast) or OpenRouter (more models)
- 🔍 **Web Search**: Tool-capable models can search the web using Brave Search API
- 💬 **Conversation Memory**: Maintains chat history per session ID
- 🎯 **Smart Tool Detection**: Automatically detects which models support web search
- 📊 **Visual Indicators**: Shows ✅/⚠️ badges for search capability status

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API Keys:
  - **Groq API Key** (get from [console.groq.com](https://console.groq.com))
  - **OpenRouter API Key** (get from [openrouter.ai/keys](https://openrouter.ai/keys))
  - **Brave Search API Key** (get from [api.search.brave.com](https://api.search.brave.com/app/dashboard)) — **Free tier: 2,000 searches/month**

### Installation

**1. Clone this repository:**
```bash
git clone https://github.com/whtisusername/True-ly-yours-Bot.git
cd True-ly-yours-Bot
git checkout patch-1
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables:**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
BRAVE_API_KEY=your_brave_key_here
```

**4. Run the app:**
```bash
streamlit run code.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🔧 How It Works

### Architecture

The app uses **two different execution modes** depending on the selected model:

#### 1️⃣ **Agent Mode (with Web Search)** ✅

For tool-capable models:
```
User Input → Agent Decides → Need Search?
  ├─ YES → Call brave_search() → Get Results → Include in Response
  └─ NO  → Direct Response
```

#### 2️⃣ **Basic Mode (No Tools)** ⚠️

For models without tool support:
```
User Input → Prompt Template → LLM → String Output
```

### Tool-Capable Models

| Provider | Model | Web Search |
|---|---|---|
| **Groq** | `llama-3.3-70b-versatile` | ✅ |
| **Groq** | `gemma2-9b-it` | ✅ |
| **Groq** | `mixtral-8x7b-32768` | ❌ |
| **OpenRouter** | `meta-llama/llama-3.3-70b-instruct:free` | ✅ |
| **OpenRouter** | `google/gemma-2-9b-it:free` | ❌ |
| **OpenRouter** | `nvidia/llama-3.1-nemotron-70b-instruct:free` | ❌ |

---

## 📖 Usage Guide

### Selecting a Model

1. Open the **sidebar** (⚙️ Settings)
2. Choose your **AI Provider** (Groq or OpenRouter)
3. Select a **model** from the dropdown
4. Check for the status badge:
   - ✅ **"Web search enabled"** = Model can use Brave Search
   - ⚠️ **"This model doesn't support web search"** = Basic mode only

### Managing Conversations

- **Session ID**: Each unique ID maintains separate conversation memory
- **Change Session ID** to start a fresh conversation
- **Clear Memory** button erases current session history
- Message count displayed in sidebar

### Example Queries

**Without Web Search:**
```
"Explain quantum computing"
"Write a poem about Dubai"
```

**With Web Search (tool-capable models only):**
```
"What's the current weather in Dubai?"
"Latest news about AI developments in 2026"
"Who won the Formula 1 championship this year?"
```

When a tool-capable model detects it needs current information, it will automatically search the web and cite sources in its response.

---

## 🛠️ Technical Details

### File Structure

```
True-ly-yours-Bot/
├── code.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .env                 # Your API keys (gitignored)
└── README.md            # This file
```

### Key Components

**`code.py` Architecture:**

1. **`brave_search(query: str)`** - LangChain `@tool` decorated function that:
   - Accepts search query from the AI agent
   - Calls Brave Search API
   - Returns top 5 formatted results with titles, descriptions, and URLs

2. **`TOOL_CAPABLE_MODELS`** - Dictionary mapping providers to models that support function calling

3. **`create_chat_chain(llm, session_id, use_tools)`** - Creates either:
   - `AgentExecutor` with Brave Search tool (if `use_tools=True`)
   - Basic chain with `StrOutputParser` (if `use_tools=False`)

4. **`RunnableWithMessageHistory`** - Wraps the chain to automatically load/save conversation history per session

### Dependencies

```txt
streamlit              # Web UI framework
langchain              # LLM orchestration
langchain-groq         # Groq provider
langchain-openai       # OpenRouter/OpenAI provider
langchain-community    # Chat message history
python-dotenv          # Environment variables
requests               # HTTP library for Brave Search API
```

---

## 🔒 Security & Privacy

- **API Keys**: Never commit `.env` file to version control (already in `.gitignore`)
- **Brave Search**: Free tier rate limits apply (2,000 queries/month)
- **Conversation Data**: Stored in memory only, cleared on app restart
- **No Data Collection**: All data stays local except API calls to chosen providers

---

## 🐛 Troubleshooting

### "GROQ_API_KEY not found!"

**Solution:** Check that `.env` file exists and contains your API key:
```bash
cat .env  # Should show GROQ_API_KEY=gsk_...
```

### "Brave Search API key not configured"

**Solution:** Add `BRAVE_API_KEY` to your `.env` file. The app will still work but web search won't be available.

### Model shows ⚠️ even though I want search

**Solution:** Switch to a tool-capable model (see table above). Only certain models support function calling.

### Rate limit errors

**Solution:**
- **Groq**: Wait 60 seconds (very generous free tier)
- **Brave Search**: Free tier = 2,000/month, paid tiers available
- **OpenRouter**: Depends on model, some are pay-per-use

---

## 🤝 Contributing

This is a fork with web search enhancements. To contribute:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

---

## 📄 License

MIT License - See original repository for full license text.

---

## 🙏 Credits

- **Original Project**: [viswa-fataniya/True-ly-yours-Bot](https://github.com/viswa-fataniya/True-ly-yours-Bot)
- **Web Search Enhancement**: Added by [whtisusername](https://github.com/whtisusername)
- **Powered By**:
  - [Groq](https://groq.com) - Fast LLM inference
  - [OpenRouter](https://openrouter.ai) - Multi-model access
  - [Brave Search API](https://brave.com/search/api/) - Privacy-focused search
  - [LangChain](https://langchain.com) - LLM framework
  - [Streamlit](https://streamlit.io) - Web app framework

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

---

**Made with ❤️ in Dubai** 🇦🇪
