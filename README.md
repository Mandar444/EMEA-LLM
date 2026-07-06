# 🔮 EMEA Offline Intelligent Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An Offline Intelligent Assistant designed to parse, index, and answer questions from technical manual documentations. Currently pre-configured with the **MySAF-T User Guide**. Built to run entirely offline on standard CPU hardware with zero external API calls or network requests, securing absolute data privacy.

---

## 🚀 Key Features

* **Hybrid Search Pipeline**: Seamlessly merges lexical retrieval (BM25) with deep semantic vector space scoring (Siamese Bi-LSTM) using a weighted average.
* **Extractive Sentence Scorer & Filter**: Tokenizes text chunks into sentences, scores candidate relevance, filters unrelated sentences using query keywords, and yields high-precision answers.
* **Smart Intent Formatter**: Formats answers into structured layouts depending on the query type (e.g., numbered step lists for "How-to", tables for comparison, and bullet points for lists).
* **In-Context Contradiction Warning**: Analyzes overlapping candidate facts from different sections to automatically flag conflicting instructions or documentation discrepancies.
* **Automated Evaluation Suite**: Robust diagnostics assessing retrieval accuracy, precision, recall, F1, hallucination rate, and citation correctness on 100+ cases.
* **Premium Glassmorphic Dashboard UI**: Clean React dashboard featuring real-time training controls, doc manager, and active citation mapping.

---

## 📦 Folder Structure

```directory
├── backend/
│   ├── api/                 # FastAPI routes for Chat, Documents, & Training
│   ├── core/                # Database connections, Document parser, & Cleaners
│   ├── db/                  # SQLite models & migrations
│   ├── ml/                  # BM25, Word2Vec, Siamese Bi-LSTM, & Evaluator
│   ├── main.py              # Backend service entrypoint
│   └── Dockerfile           # Backend image builder
├── frontend/
│   ├── src/
│   │   ├── services/        # API service clients
│   │   ├── App.tsx          # Main React Application
│   │   └── index.css        # Glassmorphic responsive styling
│   └── Dockerfile           # Nginx server image builder
├── storage/                 # Shared local volume for model checkpoints & db
└── docker-compose.yml       # Orchestrates frontend & backend containers
```

---

## 🛠️ Prerequisites

To run this application, make sure you have installed:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* *Alternatively (for local running)*: Python 3.10+ and Node.js 18+

---

## 🐳 Docker Setup

The easiest way to boot the full ecosystem is using Docker Compose:

1. Clone the repository and navigate to the directory:
   ```bash
   cd "EMEAit llm"
   ```
2. Build and start the services:
   ```bash
   docker-compose up --build
   ```
3. Open your browser:
   * **Frontend App**: [http://localhost:3000](http://localhost:3000)
   * **Backend API Swagger docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚡ Quick Start (Local Setup)

If you prefer to run the components locally without Docker:

### 1. Run the Backend
```bash
# Navigate to backend
cd backend

# Create a virtual environment and activate
python -m venv venv
venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run main FastAPI server
python -m backend.main
```

### 2. Run the Frontend
```bash
# Navigate to frontend
cd frontend

# Install packages
npm install

# Run the dev server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📊 Run Automated Quality Evaluation

To run the evaluation framework and check retrieval accuracy, precision, recall, and F1 scores on the 100+ cases dataset:

```bash
# Set Python path to workspace root
$env:PYTHONPATH="c:\Users\smand\OneDrive\Desktop\EMEAit llm"

# Execute evaluator
python -m backend.ml.evaluator
```

---

## 🔗 Documentation Links

<details>
  <summary>📄 Click to expand local resources & files</summary>

  - [Backend Configuration](file:///c:/Users/smand/OneDrive/Desktop/EMEAit%20llm/backend/config.py)
  - [Inference Coordinator Pipeline](file:///c:/Users/smand/OneDrive/Desktop/EMEAit%20llm/backend/ml/inference.py)
  - [Answer Generation & Formatting Layer](file:///c:/Users/smand/OneDrive/Desktop/EMEAit%20llm/backend/ml/answer_generator.py)
  - [Automated Evaluator Module](file:///c:/Users/smand/OneDrive/Desktop/EMEAit%20llm/backend/ml/evaluator.py)
  - [Frontend Styling Sheet](file:///c:/Users/smand/OneDrive/Desktop/EMEAit%20llm/frontend/src/index.css)
</details>

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

For anything Reach out to 
Mandar Shinde : mandarshinde627@gmail.com

