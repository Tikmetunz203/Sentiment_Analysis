# Thai Restaurant Sentiment Analysis
> RAG + Few-shot Learning ด้วย Typhoon LLM และ Wongnai Dataset

---

## Overview

ระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) สำหรับรีวิวร้านอาหารภาษาไทย โดยใช้เทคนิค **Retrieval-Augmented Generation (RAG)** ร่วมกับ **Few-shot Learning** และ **Large Language Model (LLM)** เพื่อจัดประเภทรีวิวออกเป็น 3 ระดับ

| Label | ความหมาย |
|---|---|
| Positive | รีวิวเชิงบวก |
| Neutral | รีวิวกลางๆ หรือ mixed |
| Negative | รีวิวเชิงลบ |

---

## Features

- รองรับ **ภาษาถิ่น** — เหนือ (ลำ, ลำขนาด) อีสาน (แซ่บ, แซ่บอีหลี) ใต้ (หรอย, หรอยจังฮู้)
- ตรวจจับ **คำประชด** (Sarcasm) เช่น "บริการดีมากสั่งชาตินี้ได้กินชาติหน้า"
- **Highlight คำ** ที่มีผลต่อ sentiment แยกสี Positive / Neutral / Negative
- แสดง **Retrieved Context** จาก dataset ที่ใกล้เคียงที่สุดพร้อม similarity score
- **Confidence Chart** แสดงความมั่นใจของ model ในแต่ละ class

---

## System Architecture

```
Input Review
     │
     ▼
Sentence Embedding (multilingual-e5-small)
     │
     ▼
Vector Retrieval — Cosine Similarity → Top-2 (threshold ≥ 0.5)
     │
     ▼
RAG Prompt Builder — Dialect rules + Sarcasm rules + Context
     │
     ▼
LLM Inference (Typhoon 2.5 via Ollama) — temp=0, JSON mode
     │
     ▼
JSON Output — sentiment · confidence_scores · keywords
     │
     ▼
Gradio UI — Highlight · Pie Chart · Retrieved Context
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/username/thai-sentiment-rag.git
cd thai-sentiment-rag
```

### 2. Install Python Dependencies

```bash
pip install gradio pandas sentence-transformers scikit-learn numpy requests matplotlib
```

### 3. Install Ollama

โหลดจาก [ollama.com](https://ollama.com) แล้วติดตั้ง จากนั้น pull model

```bash
ollama pull scb10x/typhoon2.5-qwen3-4b
```

---

## Dataset

ใช้ข้อมูลจาก **Wongnai Dataset** รีวิวร้านอาหารภาษาไทย โดย concat คอลัมน์ `food_sentiment` และ `service_sentiment` เป็นข้อความเดียว แล้วใช้ทั้งหมดเป็น Vector Database สำหรับ Retrieval

```
📁 โปรเจ็ก/
├── sentiment_rag.py
└── Review_Raw_Data - review_to_csv.csv   ← ต้องมีไฟล์นี้
```

---

## Usage

```bash
# Terminal 1 — เปิด Ollama
ollama serve

# Terminal 2 — รันโปรแกรม
python sentiment_rag.py
```

เปิด browser ไปที่ `http://127.0.0.1:7860`

---

## Project Structure

```
📁 thai-sentiment-rag/
├── sentiment_rag.py                      # Main app
├── README.md
└── Review_Raw_Data - review_to_csv.csv   # Wongnai Dataset
```

---

## Key Techniques

| เทคนิค | รายละเอียด |
|---|---|
| RAG | ดึง top-2 reviews ที่ใกล้เคียงด้วย cosine similarity |
| Similarity Threshold | กรอง score < 0.5 ทิ้ง ป้องกัน irrelevant context |
| Few-shot Learning | ใส่ตัวอย่างจริงใน prompt ก่อน LLM ตัดสิน |
| Prompt Engineering | Dialect rules + Sarcasm rules + Mixed sentiment rules |
| Placeholder Technique | ป้องกัน HTML พังเวลา highlight คำซ้อนกัน |

---

## Model

| Component | Model |
|---|---|
| LLM | scb10x/typhoon2.5-qwen3-4b (via Ollama) |
| Embedding | intfloat/multilingual-e5-small |

---

## LLM Configuration

| Parameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| temperature | 0 | Deterministic output |
| format | json | บังคับ output เป็น JSON |
| num_ctx | 1024 | Context window |
| num_predict | 600 | Max output tokens |

---

## Requirements

```
gradio
pandas
sentence-transformers
scikit-learn
numpy
requests
matplotlib
ollama (local)
```
