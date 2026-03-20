"""
Thai Restaurant Sentiment Analysis — RAG + Few-shot
====================================================
"""

import json
import uuid

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Review_Raw_Data - review_to_csv.csv")

df['text'] = (
    df['food_sentiment'].fillna('').str.strip() + " " +
    df['service_sentiment'].fillna('').str.strip()
).str.strip()

df = df[df['text'] != ''][['text', 'overall']].dropna().reset_index(drop=True)

print(f"✅ Loaded: {len(df)} rows")

# ============================================================
# 2. EMBEDDING MODEL
# ============================================================

print("⏳ Loading embedding model...")
embed_model = SentenceTransformer('intfloat/multilingual-e5-small')
print(f"🚀 Running on: {embed_model.device}")

corpus_texts  = df['text'].tolist()
corpus_labels = df['overall'].tolist()

dataset_embeddings = embed_model.encode(
    corpus_texts, normalize_embeddings=True, show_progress_bar=True
)
print("✅ Embeddings ready")

# ============================================================
# 3. RETRIEVAL FUNCTION
# ============================================================

def retrieve_similar_reviews(
    query: str, top_k: int = 2, threshold: float = 0.5
) -> tuple[str, list[dict]]:

    query_vec  = embed_model.encode([query], normalize_embeddings=True)
    sim_scores = cosine_similarity(query_vec, dataset_embeddings)[0]

    sorted_idx = np.argsort(sim_scores)[::-1]
    top_idx    = [i for i in sorted_idx if sim_scores[i] >= threshold][:top_k]

    retrieved_items = []
    context_str     = ""

    for rank, i in enumerate(top_idx, 1):
        item = {
            "rank":      rank,
            "review":    corpus_texts[i],
            "sentiment": corpus_labels[i],
            "score":     float(round(sim_scores[i], 4)),
        }
        retrieved_items.append(item)
        context_str += (
            f"\nExample {rank} (similarity={item['score']:.3f}):\n"
            f"Review: {item['review']}\n"
            f"Sentiment: {item['sentiment']}\n"
        )

    return context_str, retrieved_items

# ============================================================
# 4. PROMPT BUILDER
# ============================================================

DIALECT_RULES = """
**Thai Dialect Vocabulary:**
- ลำ, ลำขนาด (Northern) = อร่อย, อร่อยมาก → Positive
- แซ่บ, แซ่บอีหลี (Isan) = อร่อย → Positive
- หรอย, หรอยจังฮู้ (Southern) = อร่อย → Positive
- บ่อร่อย, บ่ลำ (Northern/Isan) = ไม่อร่อย → Negative

**CRITICAL RULES FOR SARCASM (กฎการจับคำประชด):**
- Thais often use highly positive words sarcastically to complain
  (e.g., "บริการดีมาก สั่งชาตินี้ได้กินชาติหน้า", "อร่อยจนต้องคายทิ้ง").
- If a sentence starts positive but ends with a contradiction or complaint, it is heavily **NEGATIVE**.
- Always analyze the "True Intention" before deciding.
"""

SARCASM_RULES = """
**Sarcasm Detection Rules:**
- ถ้าขึ้นต้นด้วยคำ positive แต่จบด้วย contradiction → Negative
- ถ้ามีทั้ง positive และ negative คนละด้าน เช่น อาหารอร่อย แต่ที่จอดรถน้อย → Neutral
- คำว่า "แต่", "แค่", "นิดหน่อย" ที่ตามหลัง positive = mixed → Neutral
- วิเคราะห์ "True Intention" เสมอก่อนตัดสิน
"""

def build_prompt(review: str, context: str) -> str:
    ctx_section = (
        f"\nContext from similar reviews:\n{context}"
        if context.strip()
        else "\n(No similar reviews retrieved)"
    )

    return f"""You are a Thai Sentiment Analysis expert.
Understand Standard Thai, Slang, and Regional Dialects.
{DIALECT_RULES}
{SARCASM_RULES}

Task:
1. Classify Sentiment: Positive, Neutral, or Negative.
2. Estimate Confidence Score (must sum to 100).
3. Identify Positive / Negative / Neutral keywords.
4. Explain reasoning IN THAI only.
{ctx_section}

Input Review: "{review}"

Respond ONLY in valid JSON:
{{
    "sentiment": "Positive/Neutral/Negative — เหตุผลสั้นๆ 1 ประโยค",
    "confidence_scores": {{
        "Positive": <0-100>,
        "Neutral":  <0-100>,
        "Negative": <0-100>
    }},
    "positive_words": ["..."],
    "negative_words": ["..."],
    "neutral_words":  ["..."],
}}"""

# ============================================================
# 5. LLM INFERENCE
# ============================================================

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "scb10x/typhoon2.5-qwen3-4b"

def call_llm(prompt: str) -> dict | None:
    import time
    try:
        t0   = time.time()
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 600,
                    "num_ctx":     1024,
                },
            },
            timeout=90,
        )
        resp.raise_for_status()
        raw  = resp.json().get("response", "")
        data = json.loads(raw)
        print(f"⏱ LLM inference: {time.time()-t0:.2f}s")
        return data
    except json.JSONDecodeError:
        print(f"[LLM] JSON parse failed. Raw: {raw[:300]}")
        return None
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None

def predict(review: str) -> dict | None:
    context, retrieved = retrieve_similar_reviews(review)
    prompt  = build_prompt(review, context)
    result  = call_llm(prompt)
    if result is not None:
        result["_retrieved"] = retrieved
    return result

# ============================================================
# 6. HIGHLIGHT UTILITY
# ============================================================

def highlight_text(
    text: str,
    pos_words: list,
    neg_words: list,
    neu_words: list,
) -> str:
    if not text:
        return ""

    all_words = (
        [(w.strip(), "pos") for w in pos_words if isinstance(w, str) and w.strip()] +
        [(w.strip(), "neg") for w in neg_words if isinstance(w, str) and w.strip()] +
        [(w.strip(), "neu") for w in neu_words if isinstance(w, str) and w.strip()]
    )
    all_words.sort(key=lambda x: len(x[0]), reverse=True)

    highlighted  = text
    placeholders = {}

    for word, sentiment in all_words:
        if word not in highlighted:
            continue
        ph = f"[{uuid.uuid4().hex[:8]}]"
        placeholders[ph] = (word, sentiment)
        highlighted = highlighted.replace(word, ph)

    COLOR_MAP = {
        "pos": "background:#d4edda;color:#155724",
        "neg": "background:#f8d7da;color:#721c24",
        "neu": "background:#fff3cd;color:#856404",
    }
    for ph, (word, sentiment) in placeholders.items():
        style = COLOR_MAP[sentiment] + ";padding:2px 4px;border-radius:4px;font-weight:bold"
        highlighted = highlighted.replace(ph, f"<span style='{style}'>{word}</span>")

    return f"<div style='line-height:2;font-size:16px'>{highlighted}</div>"

# ============================================================
# 7. CHART
# ============================================================

def plot_confidence_chart(scores: dict):
    if not scores:
        return None

    labels    = list(scores.keys())
    values    = list(scores.values())
    color_map = {"Positive": "#66cc66", "Neutral": "#586d7a", "Negative": "#ff6666"}
    colors    = [color_map.get(l, "#cccccc") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 1},
    )
    for t, c in zip(texts, colors):
        t.set_color(c)
        t.set_fontsize(12)
        t.set_weight("bold")
    ax.set_title("AI Confidence", color="white", fontsize=14, weight="bold")
    ax.axis("equal")
    plt.close(fig)
    return fig

# ============================================================
# 8. GRADIO UI
# ============================================================

def predict_ui(text: str):
    if not text.strip():
        return "", "", "", None

    result = predict(text)
    if result is None:
        return "Error", text, "ไม่สามารถ parse JSON ได้", None

    label     = result.get("sentiment", "Unknown")
    pos_words = result.get("positive_words", [])
    neg_words = result.get("negative_words", [])
    neu_words = result.get("neutral_words",  [])
    scores    = result.get("confidence_scores", {})
    retrieved = result.get("_retrieved", [])

    highlighted = highlight_text(text, pos_words, neg_words, neu_words)
    fig         = plot_confidence_chart(scores)

    ctx_lines = [
        f"[{item['rank']}] (score={item['score']:.3f}, label={item['sentiment']})\n"
        f"     {item['review'][:80]}..."
        for item in retrieved
    ]
    retrieved_text = (
        "\n".join(ctx_lines) if ctx_lines
        else "ไม่มี context ที่เกี่ยวข้อง (score < threshold)"
    )

    return label, highlighted, retrieved_text, fig


with gr.Blocks(title="Thai Sentiment Dashboard (RAG + Few-shot)") as demo:
    gr.Markdown("## Thai Restaurant Sentiment Dashboard\nRAG + Few-shot · Typhoon LLM · Wongnai Dataset")

    with gr.Row():
        with gr.Column(scale=2):
            txt_input = gr.Textbox(
                lines=4,
                placeholder="เช่น อาหารอร่อย แต่รอนานมาก",
                label="รีวิว",
            )
            btn = gr.Button("วิเคราะห์ Sentiment", variant="primary")
            out_highlighted = gr.HTML(label="Highlighted Review")
            out_context     = gr.Textbox(label="ตัวอย่างที่ดึงมาจาก dataset (Retrieved Context RAG)")

        with gr.Column(scale=1):
            out_label = gr.Label(label="ผลการวิเคราะห์")
            out_chart = gr.Plot(label="Confidence Chart")

    btn.click(
        fn=predict_ui,
        inputs=txt_input,
        outputs=[out_label, out_highlighted, out_context, out_chart],
    )

    gr.Examples(
        examples=[
            ["อร่อยมาก พนักงานดีสุดๆ"],
            ["รสชาติงั้นๆ พอกินได้"],
            ["แย่มาก รอนานสุดๆ"],
            ["อาหารดีแต่บริการช้า"],
            ["บริการดีมากสั่งชาตินี้ได้กินชาติหน้า"],
            ["แซ่บอีหลี กินแล้วอยากมาอีก"],
            ["หรอยจังฮู้ แต่ที่จอดรถน้อยไปหน่อย"],
        ],
        inputs=txt_input,
    )

demo.launch()