# 🚀 Asset360 AI – Enterprise Asset Intelligence Platform

> **Transforming Engineering Documents into Intelligent Digital Assets**

Asset360 AI is an enterprise-grade AI platform that transforms engineering documents into intelligent, searchable, and continuously evolving digital assets. Instead of simply indexing PDFs, the platform builds a Digital Twin for every engineering asset by generating company manuals, maintenance history, service reports, inspection summaries, and knowledge graph relationships.

The platform combines **Retrieval-Augmented Generation (RAG)**, **Knowledge Graphs**, **OCR**, **Vision AI**, **Semantic Search**, and **Large Language Models (LLMs)** to provide engineers with accurate, citation-backed answers while preserving the complete lifecycle of enterprise assets.

---

# 🌟 Key Features

- 📄 Intelligent Document Upload
- 🤖 AI-Powered Document Understanding
- 📚 Enterprise RAG Pipeline
- 🧠 Knowledge Graph Generation
- 🏭 Asset360 Digital Twin
- 🔍 Semantic Search
- 🖼 OCR & Vision AI
- 📊 Engineering Metadata Extraction
- 🛠 Automatic Asset History Generation
- 📈 Maintenance Timeline Generation
- 📑 AI-generated Company Manuals
- 📋 Inspection Reports
- 🔧 Service Reports
- 📍 Citation-based Answers
- 🌐 Multi-document Reasoning
- ⚡ Real-time Asset Dashboard

---

# 📌 Problem Statement

Engineering organizations maintain thousands of technical documents including manuals, SOPs, inspection reports, maintenance logs, service records, and P&ID diagrams.

Finding the correct information requires engineers to manually search across multiple systems.

Asset360 AI solves this by converting every engineering document into structured enterprise knowledge and enabling natural language interaction with engineering assets.

---

# 💡 Solution Overview

Instead of treating uploaded documents as isolated files, Asset360 AI creates an intelligent Digital Twin for every engineering asset.

Whenever a new engineering document is uploaded, the platform can automatically:

- Detect engineering assets
- Extract metadata
- Generate company manuals
- Build maintenance history
- Generate service reports
- Create inspection reports
- Build asset timelines
- Update the Knowledge Graph
- Refresh Asset360 Dashboard

---

# 🏗 System Architecture

```
                Engineering Documents
                        │
                        ▼
               Document Upload API
                        │
                        ▼
             OCR + Vision Processing
                        │
                        ▼
             Metadata Extraction
                        │
                        ▼
            Engineering Entity Detection
                        │
                        ▼
              Semantic Chunking
                        │
                        ▼
             Embedding Generation
                        │
                        ▼
              Vector Database (RAG)
                        │
                        ▼
         Asset History Generation Engine
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
 Company Manual   Maintenance Log   Timeline
        │               │               │
        └───────────────┼───────────────┘
                        ▼
               Knowledge Graph
                        │
                        ▼
               Asset360 Dashboard
                        │
                        ▼
             Citation-based AI Chat
```

---

# ⚙ Technology Stack

## Frontend

- React
- TypeScript
- TailwindCSS
- Vite

## Backend

- FastAPI
- Python

## AI

- Hugging Face Models
- OpenAI Compatible Models
- Sentence Transformers

## Retrieval

- FAISS / ChromaDB

## Vision

- OCR
- Vision Language Models

## Graph

- NetworkX
- Knowledge Graph

---

# 📂 Project Structure

```text
Asset360-AI/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── layouts/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── contexts/
│   │   └── App.tsx
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── retrieval/
│   ├── ingestion/
│   ├── graph/
│   ├── history/
│   ├── ocr/
│   ├── vision/
│   ├── llm/
│   ├── embeddings/
│   ├── utils/
│   ├── main.py
│   └── requirements.txt
│
├── data/
│   ├── uploads/
│   ├── manuals/
│   ├── sop/
│   ├── pid/
│   ├── vectors/
│   ├── metadata/
│   ├── assets/
│   │
│   ├── P-101/
│   │   ├── company_manual.md
│   │   ├── maintenance_report.md
│   │   ├── inspection_report.md
│   │   ├── service_report.md
│   │   ├── asset_history.md
│   │   ├── executive_summary.md
│   │   ├── metadata.json
│   │   ├── graph.json
│   │   └── timeline.md
│   │
│   └── C-205/
│       └── ...
│
├── docs/
│   ├── Architecture.pdf
│   ├── Solution_Document.pdf
│   ├── API_Documentation.md
│   └── Screenshots/
│
├── scripts/
│
├── tests/
│
├── docker/
│
├── README.md
│
└── LICENSE
```

---

# 🔄 Workflow

```
Upload Document
        │
        ▼
OCR & Vision Processing
        │
        ▼
Metadata Extraction
        │
        ▼
Asset Detection
        │
        ▼
Generate Asset History?
        │
   ┌────┴────┐
   │         │
 No         Yes
   │         │
Index PDF    │
             ▼
Generate Manuals
Generate Reports
Generate Timeline
Update Graph
Refresh Asset360
```

---

# 🏭 Asset360 Digital Twin

Every engineering asset maintains:

- Equipment Profile
- Company Manual
- OEM Manual
- Maintenance Reports
- Inspection Reports
- Service Reports
- Timeline
- Metadata
- Knowledge Graph
- Related Assets
- P&ID References
- Health Score

---

# 🤖 AI Generated Documents

After uploading an engineering document, Asset360 AI can automatically generate:

- Company Manual
- Maintenance Report
- Inspection Report
- Service Report
- Asset Timeline
- Executive Summary
- Preventive Maintenance Guide
- Failure Analysis
- Root Cause Analysis
- Asset Metadata

---

# 🔍 Enterprise Search

Users can ask questions such as:

- Explain Pump P-101.
- Show maintenance history.
- List all inspections.
- What caused the last failure?
- Which equipment is connected?
- Generate troubleshooting steps.
- Summarize this manual.

Every answer is generated using Retrieval-Augmented Generation (RAG) and includes citations to the original engineering documents.

---

# 🚀 Future Enhancements

- IoT Sensor Integration
- Predictive Maintenance
- SAP Integration
- IBM Maximo Integration
- CMMS Connectivity
- Digital Twin Analytics
- Real-time Monitoring
- 3D Asset Visualization

---

# 👥 Team 

bs22b032 - Economics Times Hackathon 2026

Asset360 AI – Enterprise Asset Intelligence Platform

---

# 📄 License

This project is developed for the Economics Times Hackathon 2026.
