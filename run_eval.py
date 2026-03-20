import pandas as pd
import requests
import json
from sklearn.metrics import accuracy_score, classification_report

# 1. โหลดข้อมูลเทส (สุ่มมา 50 ข้อความพอ จะได้ไม่รอนาน)
df = pd.read_csv("Review_Raw_Data - review_to_csv.csv")
df['text'] = (df['food_sentiment'].fillna('') + " " + df['service_sentiment'].fillna('')).str.strip()
df = df[df['text'] != ''][['text', 'overall']].dropna()

# สุ่มมา 50 ตัวอย่าง
test_df = df.sample(n=50, random_state=42)

# 2. ฟังก์ชันยิง API แบบไม่มี RAG (เอาแค่ Prompt เน้นๆ)
def get_prediction(review):
    prompt = f"""You are a Thai Sentiment Analysis expert.
Understand Standard Thai, Slang, and Regional Dialects.
**Sarcasm Detection Rules:**
- If a sentence starts positive but ends with a contradiction or complaint, it is heavily **NEGATIVE**.
Classify the Sentiment of this review: "{review}"
Respond ONLY in valid JSON: {{"sentiment": "Positive/Neutral/Negative"}}"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "scb10x/typhoon2.5-qwen3-4b", "prompt": prompt, "stream": False, "format": "json"},
            timeout=30
        )
        data = json.loads(resp.json()["response"])
        return data.get("sentiment", "Unknown")
    except:
        return "Unknown"

# 3. รันเก็บตัวเลข
y_true = test_df['overall'].tolist()
y_pred = []

print("กำลังรันประเมินผล 50 ข้อความ (รอประมาณ 1-2 นาที)...")
for i, text in enumerate(test_df['text']):
    pred = get_prediction(text)
    y_pred.append(pred)
    print(f"[{i+1}/50] อารมณ์จริง: {y_true[i]:<10} | AI ทาย: {pred}")

# 4. สรุปผลเป็นตัวเลข!
print("\n" + "="*40)
print(f"✅ Accuracy (ความแม่นยำ): {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("="*40)
print(classification_report(y_true, y_pred, zero_division=0))