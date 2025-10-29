







1. README.md - GitHub Repository
text
# 🏦 QUESTRAG - Banking QUEries and Support system via Trained Reinforced RAG

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3.1-blue.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An intelligent banking chatbot powered by **Retrieval-Augmented Generation (RAG)** and **Reinforcement Learning (RL)** to provide accurate, context-aware responses to Indian banking queries while optimizing token costs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Performance Metrics](#performance-metrics)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

QUESTRAG is an **advanced banking chatbot** designed to revolutionize customer support in the Indian banking sector. By combining **Retrieval-Augmented Generation (RAG)** with **Reinforcement Learning (RL)**, the system intelligently decides when to fetch external context from a knowledge base and when to respond directly, **reducing token costs by up to 31%** while maintaining high accuracy.

### Problem Statement

Existing banking chatbots suffer from:
- ❌ Limited response flexibility (rigid, rule-based systems)
- ❌ Poor handling of informal/real-world queries
- ❌ Lack of contextual understanding
- ❌ High operational costs due to inefficient token usage
- ❌ Low user satisfaction and trust

### Solution

QUESTRAG addresses these challenges through:
- ✅ **Domain-specific RAG** trained on RBI regulations and banking policies
- ✅ **RL-optimized policy network** (BERT-based) for smart context-fetching decisions
- ✅ **Fine-tuned retriever model** using InfoNCE + Triplet Loss
- ✅ **Automated response evaluation** with reward-based learning
- ✅ **Full-stack web application** with modern UI/UX

---

## 🌟 Key Features

### 🤖 Intelligent RAG Pipeline
- **FAISS-powered retrieval** for fast similarity search across 19,352+ banking documents
- **Fine-tuned embedding model** (`e2-base-v5`) trained on English + Hinglish paraphrases
- **Semantic search** with adjustable similarity thresholds
- **Context-aware response generation** using Gemini 2.0 Flash Lite

### 🧠 Reinforcement Learning System
- **BERT-based policy network** (`bert-base-uncased`) for FETCH/NO_FETCH decisions
- **Reward-driven optimization**:
  - +2.0 for accurate direct responses (NO_FETCH)
  - +0.5 for necessary context fetching (FETCH)
  - -0.5 for incorrect direct responses
- **31% token cost reduction** through smart retrieval optimization
- **Monte Carlo Dropout** for uncertainty estimation

### 🎨 Modern Web Interface
- **React 18** with Vite for blazing-fast performance
- **Responsive design** with Tailwind CSS
- **Real-time chat interface** with typing indicators
- **Conversation history** with session management
- **JWT-based authentication** for secure access

### 🔐 Enterprise-Ready Backend
- **FastAPI** for high-performance async operations
- **MongoDB** for scalable data storage
- **JWT authentication** with role-based access control
- **Comprehensive logging** for compliance and debugging
- **Rate limiting** and API key management

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────┐
│ USER INTERFACE │
│ (React + Tailwind CSS) │
└──────────────────────────┬──────────────────────────────────┘
│ HTTP/REST API
┌──────────────────────────▼──────────────────────────────────┐
│ FASTAPI BACKEND │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ CHAT SERVICE (Orchestrator) │ │
│ └─────┬──────────────────────────────┬──────────────────┘ │
│ │ │ │
│ ┌─────▼──────────┐ ┌──────▼────────────┐ │
│ │ POLICY │ │ RAG PIPELINE │ │
│ │ NETWORK │───decides──▶│ │ │
│ │ (BERT-RL) │ FETCH/ │ - Retriever │ │
│ │ │ NO_FETCH │ - FAISS Index │ │
│ └────────────────┘ │ - LLM Generator │ │
│ └───────────────────┘ │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ MONGODB (Conversations + Users) │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

text

### Workflow
1. **User Query** → FastAPI receives query via REST API
2. **Policy Decision** → BERT-based RL model decides FETCH or NO_FETCH
3. **Conditional Retrieval** → If FETCH: Retrieve top-5 docs from FAISS
4. **Response Generation** → Gemini generates response (with/without context)
5. **Evaluation & Logging** → Response evaluated, logged, and RL model updated

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 18.3.1 + Vite 5.4.2
- **Styling**: Tailwind CSS 3.4.1
- **State Management**: React Context API
- **HTTP Client**: Axios 1.6.5
- **Routing**: React Router DOM 6.21.3
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI 0.104.1
- **Database**: MongoDB 4.5.0 + Motor (async driver)
- **Authentication**: PyJWT 2.8.0 + Passlib (bcrypt)
- **ML/AI**:
  - PyTorch 2.1.2 (Policy Network)
  - Transformers 4.36.2 (BERT models)
  - Sentence-Transformers 2.2.2 (Embeddings)
  - FAISS-CPU 1.7.4 (Vector search)
  - LangChain 0.1.0 (LLM orchestration)
- **LLM APIs**:
  - Google Gemini 2.0 Flash Lite (Response generation)
  - Groq (Optional - for evaluation)

### DevOps & Tools
- **Environment Management**: Python `venv` + npm
- **Configuration**: `python-dotenv` for env variables
- **CORS**: FastAPI middleware
- **Logging**: Python `logging` module
- **Version Control**: Git

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB 6.0+
- Google Gemini API key

### 1. Clone Repository
git clone https://github.com/yourusername/questrag.git
cd questrag

text

### 2. Backend Setup
cd backend

Create virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Create .env file
cp .env.example .env

Edit .env with your API keys and MongoDB URI
text

### 3. Frontend Setup
cd frontend

Install dependencies
npm install

Create .env file
cp .env.example .env

Edit .env with backend URL
text

### 4. Download Pre-trained Models
Download trained models from releases
Place in backend/app/models/
- best_policy_model.pth
- best_retriever_model.pth
- faiss_index.pkl
text

---

## ⚙️ Configuration

### Backend Configuration (`backend/.env`)

Environment
ENVIRONMENT=development
DEBUG=True

MongoDB
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=questrag_db

JWT Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

Google Gemini API
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-lite
GEMINI_REQUESTS_PER_MINUTE=60

Optional: Groq API
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama3-70b-8192
GROQ_REQUESTS_PER_MINUTE=30

Model Paths
POLICY_MODEL_PATH=app/models/best_policy_model.pth
RETRIEVER_MODEL_PATH=app/models/best_retriever_model.pth
FAISS_INDEX_PATH=app/models/faiss_index.pkl
KB_PATH=app/data/final_knowledge_base.jsonl

Device Settings
DEVICE=cpu # or 'cuda' for GPU

LLM Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

RAG Parameters
TOP_K=5
SIMILARITY_THRESHOLD=0.5
MAX_CONTEXT_LENGTH=2000

Policy Network Parameters
POLICY_MAX_LEN=256
CONFIDENCE_THRESHOLD=0.7

CORS
ALLOWED_ORIGINS=http://localhost:5173

text

### Frontend Configuration (`frontend/.env`)

VITE_API_URL=http://localhost:8000

text

---

## 🚀 Usage

### Start Backend Server
cd backend
source venv/bin/activate # Windows: venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

text

Backend will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Start Frontend Development Server
cd frontend
npm run dev

text

Frontend will be available at: `http://localhost:5173`

### Access Application
1. Open `http://localhost:5173` in your browser
2. Register a new account or login
3. Start chatting with the banking assistant!

---

## 📁 Project Structure

questrag/
│
├── backend/
│ ├── app/
│ │ ├── api/
│ │ │ └── v1/
│ │ │ ├── auth.py # Authentication endpoints
│ │ │ └── chat.py # Chat endpoints
│ │ ├── core/
│ │ │ ├── llm_manager.py # LLM orchestration
│ │ │ └── security.py # JWT & password hashing
│ │ ├── ml/
│ │ │ ├── policy_network.py # BERT-based RL policy
│ │ │ └── retriever.py # Fine-tuned retriever
│ │ ├── models/
│ │ │ ├── best_policy_model.pth
│ │ │ ├── best_retriever_model.pth
│ │ │ └── faiss_index.pkl
│ │ ├── repositories/
│ │ │ ├── user_repository.py # User CRUD operations
│ │ │ └── conversation_repo.py # Conversation storage
│ │ ├── services/
│ │ │ └── chat_service.py # Main chat orchestration
│ │ ├── config.py # Settings management
│ │ └── main.py # FastAPI app entry
│ ├── requirements.txt
│ └── .env
│
└── frontend/
├── src/
│ ├── components/
│ │ ├── Chatbox.jsx # Chat interface
│ │ ├── MessageBubble.jsx # Message display
│ │ ├── MessageInput.jsx # Input field
│ │ ├── Navbar.jsx # Navigation
│ │ └── Sidebar.jsx # Conversation history
│ ├── context/
│ │ └── AuthContext.jsx # Auth state management
│ ├── pages/
│ │ ├── Login.jsx # Login page
│ │ ├── Register.jsx # Registration page
│ │ └── Chat.jsx # Main chat page
│ ├── services/
│ │ └── api.js # API client
│ ├── App.jsx # App entry
│ └── main.jsx # React entry
├── package.json
└── .env

text

---

## 📊 Datasets

### 1. Final Knowledge Base
- **Size**: 19,352 question-answer pairs
- **Categories**: 15 banking categories
- **Intents**: 22 unique intents (focused on ATM, CARD, LOAN, ACCOUNT)
- **Source**: Combination of:
  - Bitext Retail Banking Dataset (Hugging Face)
  - RetailBanking-Conversations Dataset
  - Manually curated FAQs from SBI, ICICI, HDFC, Yes Bank, Axis Bank

### 2. Retriever Training Dataset
- **Size**: 11,655 paraphrases
- **Source**: 1,665 unique FAQs from knowledge base
- **Paraphrases per FAQ**:
  - 4 English paraphrases
  - 2 Hinglish paraphrases
  - Original FAQ
- **Training**: InfoNCE Loss + Triplet Loss

### 3. Policy Network Training Dataset
- **Size**: 182 queries from 6 chat sessions
- **Format**: (state, action, reward) tuples
- **Actions**: FETCH (1) or NO_FETCH (0)
- **Rewards**: +2.0, +0.5, -0.5

---

<!-- ## 📈 Performance Metrics

### Token Cost Optimization
┌─────────────────────┬───────────────┬────────────┐
│ Configuration │ Token Cost │ Accuracy │
├─────────────────────┼───────────────┼────────────┤
│ Always FETCH │ 149,064 │ 43% │
│ SimThr + Policy RL │ 131,136 │ 58% │
│ Policy Only (RL) │ 119,269 │ 66% │
│ SimThr Only │ 55,797 │ 58% │
└─────────────────────┴───────────────┴────────────┘

text

**Key Findings**:
- ✅ **31% token cost reduction** with RL-optimized policy
- ✅ **76% accuracy** with Policy-Only (RL) approach
- ✅ **Similarity Threshold** reduces costs drastically but limits flexibility

--- 
-->

## 📚 API Documentation

### Authentication

#### Register
POST /api/v1/auth/register
Content-Type: application/json

{
"username": "john_doe",
"email": "john@example.com",
"password": "securepassword123"
}

text

#### Login
POST /api/v1/auth/login
Content-Type: application/json

{
"username": "john_doe",
"password": "securepassword123"
}

Response:
{
"access_token": "eyJhbGciOiJIUzI1NiIs...",
"token_type": "bearer"
}

text

### Chat

#### Send Message
POST /api/v1/chat/
Authorization: Bearer <token>
Content-Type: application/json

{
"query": "What are the interest rates for home loans?",
"session_id": "optional-session-id"
}

Response:
{
"response": "Current home loan interest rates range from 8.5% to 9.5% per annum...",
"session_id": "abc123",
"metadata": {
"policy_action": "FETCH",
"retrieval_score": 0.89,
"documents_retrieved": 5
}
}

text

#### Get Conversation History
GET /api/v1/chat/history/{session_id}
Authorization: Bearer <token>

Response:
{
"session_id": "abc123",
"messages": [
{
"role": "user",
"content": "What are the interest rates?",
"timestamp": "2025-10-29T10:30:00Z"
},
{
"role": "assistant",
"content": "Current rates are...",
"timestamp": "2025-10-29T10:30:05Z"
}
]
}

text

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint + Prettier for JavaScript/React
- Write comprehensive docstrings and comments
- Add unit tests for new features
- Update documentation accordingly

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research Inspiration
- **Main Paper**: "Optimizing Retrieval Augmented Generation for Domain-Specific Chatbots with Reinforcement Learning" (AAAI 2024)
- **Additional References**:
  - "Evaluating BERT-based Rewards for Question Generation with RL"
  - "Self-Reasoning for Retrieval-Augmented Language Models"

### Open Source Resources
- [RL-Self-Improving-RAG](https://github.com/subrata-samanta/RL-Self-Improving-RAG)
- [ARENA](https://github.com/ren258/ARENA)
- [RAGTechniques](https://github.com/NirDiamant/RAGTechniques)
- [Financial-RAG-From-Scratch](https://github.com/cse-amarjeet/Financial-RAG-From-Scratch)

### Datasets
- [Bitext Retail Banking Dataset](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset)
- [RetailBanking-Conversations](https://huggingface.co/datasets/oopere/RetailBanking-Conversations)

<!-- ### Institution
- **Institute for Development and Research in Banking Technology (IDRBT), Hyderabad**
- **Guided by**: Dr. Mridula Verma
-->
---

## 📞 Contact

**Eeshanya Amit Joshi**
- Email: eeshanyajoshi@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **API Documentation**: `http://localhost:8000/docs` (when running locally)
- **Research Paper**: [AAAI 2024 Workshop](https://arxiv.org/abs/2401.06800)

---

<p align="center">Made with ❤️ for the Banking Industry</p>
