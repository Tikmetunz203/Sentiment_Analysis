
Thai Restaurant Sentiment Analysis

RAG + Few-shot Learning ด้วย Typhoon LLM และ Wongnai Dataset


Overview
ระบบวิเคราะห์ความรู้สึก (Sentiment Analysis) สำหรับรีวิวร้านอาหารภาษาไทย โดยใช้เทคนิค Retrieval-Augmented Generation (RAG) ร่วมกับ Few-shot Learning และ Large Language Model (LLM) เพื่อจัดประเภทรีวิวออกเป็น 3 ระดับ
LabelความหมายPositiveรีวิวเชิงบวกNeutralรีวิวกลางๆ หรือ mixedNegativeรีวิวเชิงลบ

Features

รองรับ ภาษาถิ่น — เหนือ (ลำ, ลำขนาด) อีสาน (แซ่บ, แซ่บอีหลี) ใต้ (หรอย, หรอยจังฮู้)
ตรวจจับ คำประชด (Sarcasm) เช่น "บริการดีมากสั่งชาตินี้ได้กินชาติหน้า"
Highlight คำ ที่มีผลต่อ sentiment แยกสี Positive / Neutral / Negative
แสดง Retrieved Context จาก dataset ที่ใกล้เคียงที่สุด
รองรับ Ablation Test เปรียบเทียบ RAG vs No-RAG


System Architecture
Input Review
     │
     ▼
Sentence Embedding (multilingual-e5-small)
     │
     ▼
Vector Retrieval — Cosine Similarity → Top-2 similar reviews
     │
     ▼
RAG Prompt Builder — Dialect rules + Sarcasm rules + Context
     │
     ▼
LLM Inference (Typhoon 2.5 via Ollama)
     │
     ▼
JSON Output — sentiment · confidence · keywords · reason
     │
     ▼
Gradio UI — Highlight · Chart · Retrieved Context

Installation
1. Clone Repository
bashgit clone https://github.com/username/thai-sentiment-rag.git
cd thai-sentiment-rag
2. Install Python Dependencies
bashpip install gradio pandas sentence-transformers scikit-learn numpy requests matplotlib seaborn
3. Install Ollama
โหลดจาก ollama.com แล้วติดตั้ง จากนั้น pull model
bashollama pull scb10x/typhoon2.5-qwen3-4b

Dataset
ใช้ข้อมูลจาก Wongnai Dataset รีวิวร้านอาหารภาษาไทย
📁 โปรเจ็ก/
├── sentiment_rag_improved.py
├── run_eval.py
└── Review_Raw_Data - review_to_csv.csv   ← ต้องมีไฟล์นี้
SplitจำนวนTrain (80%)~269 rowsTest (20%)~67 rowsLabel: Positive135 rowsLabel: Negative147 rowsLabel: Neutral54 rows

Usage
รัน Gradio UI
bash# Terminal 1 — เปิด Ollama
ollama serve

# Terminal 2 — รันโปรแกรม
python sentiment_rag_improved.py
เปิด browser ไปที่ http://127.0.0.1:7860
รัน Evaluation & Ablation Test
bashpython run_eval.py
ผลลัพธ์ที่ได้
==================================================
ABLATION TEST: RAG vs No-RAG
==================================================
With    RAG (few-shot) → Accuracy: x.xxxx | Macro F1: x.xxxx
Without RAG (zero-shot)→ Accuracy: x.xxxx | Macro F1: x.xxxx
RAG improvement        → Accuracy: +x.xxxx | F1: +x.xxxx

Project Structure
📁 thai-sentiment-rag/
├── sentiment_rag_improved.py   # Main app — RAG + Gradio UI
├── run_eval.py                 # Evaluation & Ablation Test
├── README.md
└── Review_Raw_Data - review_to_csv.csv

Key Techniques
เทคนิครายละเอียดRAGดึง top-2 reviews ที่ใกล้เคียงด้วย cosine similarityFew-shot Learningใส่ตัวอย่างจริงใน prompt ก่อน LLM ตัดสินSimilarity Thresholdกรอง score < 0.5 ทิ้ง ป้องกัน irrelevant contextPrompt EngineeringDialect + Sarcasm + Mixed sentiment rulesTrain/Test Split80/20 seed=42 ให้ผล reproducibleAblation Testเปรียบเทียบ RAG vs No-RAG พิสูจน์ว่า RAG ช่วยจริง

Model
ComponentModelLLMscb10x/typhoon2.5-qwen3-4b (via Ollama)Embeddingintfloat/multilingual-e5-small

Requirements
gradio
pandas
sentence-transformers
scikit-learn
numpy
requests
matplotlib
seaborn
ollama (local)
