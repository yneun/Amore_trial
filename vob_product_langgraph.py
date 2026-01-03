import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import easyocr

# ------------------------------
# CONFIG (Colab 친화)
# ------------------------------
MODEL_A = "microsoft/phi-3-mini-4k-instruct"
MODEL_B = "stabilityai/stable-code-instruct-3b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------------------
# MODEL LOADING (FP16, VRAM 절약)
# ------------------------------
tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)
model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_A,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B,
    device_map="auto",
    torch_dtype=torch.float16
)

def run_llm(prompt, model, tokenizer, max_new_tokens=400):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ------------------------------
# OCR (CPU로 안전하게)
# ------------------------------
ocr_reader = easyocr.Reader(["en"], gpu=False)

def run_ocr(image_path: str) -> List[str]:
    if Path(image_path).exists():
        out = ocr_reader.readtext(image_path)
        return [t for _, t, _ in out]
    return []

# ------------------------------
# LLM-A : IMAGE ROLE CLASSIFICATION
# ------------------------------
def classify_image_role(ocr_texts: List[str]) -> str:
    text = " | ".join(ocr_texts) if ocr_texts else "NO_TEXT"
    prompt = f"""
Classify the PRIMARY role of the following content
in an e-commerce product detail page.

CONTENT:
{text}

Choose ONE:
- aesthetic
- explanatory
- evidential

Return JSON only.
"""
    raw = run_llm(prompt, model_a, tokenizer_a, 120)
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}")+1])["role"]
    except:
        return "aesthetic"

# ------------------------------
# LLM-B : TEXT-ONLY EXPLICITNESS
# ------------------------------
def judge_text_explicitness(slot: str, texts: List[str], product: str):
    prompt = f"""
You are simulating a careful e-commerce consumer.

PRODUCT:
{product}

TOPIC:
{slot}

TEXT INFORMATION:
{texts}

TASK:
1. List realistic questions or complaints that could arise
   based ONLY on this text.
2. Mark each as:
   - prevented
   - partially_prevented
   - not_prevented
3. Rate how explicit the text is.

Return JSON:
{{
  "anticipated_questions": [
    {{"question": "...", "status": "prevented|partially_prevented|not_prevented"}}
  ],
  "text_explicitness": 0.0
}}
"""
    raw = run_llm(prompt, model_b, tokenizer_b)
    return json.loads(raw[raw.index("{"):raw.rindex("}")+1])

# ------------------------------
# LLM-B : IMAGE EXPLICITNESS
# ------------------------------
def judge_image_explicitness(slot: str, images: List[Dict], product: str):
    image_block = ""
    for img in images:
        image_block += f"""
- role: {img.get("role","NA")}
- OCR: {img.get("ocr_text","")}
"""
    prompt = f"""
You are simulating a careful e-commerce consumer.

PRODUCT:
{product}

TOPIC:
{slot}

IMAGE INFORMATION:
{image_block}

TASK:
1. List realistic questions or complaints that could arise
   based ONLY on these images.
2. Mark each as:
   - prevented
   - partially_prevented
   - not_prevented
3. Rate how explicit the images are.

Return JSON:
{{
  "anticipated_questions": [
    {{"question": "...", "status": "prevented|partially_prevented|not_prevented"}}
  ],
  "image_explicitness": 0.0
}}
"""
    raw = run_llm(prompt, model_b, tokenizer_b)
    return json.loads(raw[raw.index("{"):raw.rindex("}")+1])

# ------------------------------
# EXPLICITNESS MERGE
# ------------------------------
def merge_explicitness(text_e: float, image_e: float) -> float:
    base = max(text_e, image_e)
    bonus = 0.15 * min(text_e, image_e)
    return min(1.0, base + bonus)

# ------------------------------
# PIPELINE : CSV 기반
# ------------------------------

def run_pipeline_from_csv_with_images(csv_path: str, product_name: str):
    df = pd.read_csv(csv_path, encoding="latin1")  # 인코딩 주의

    data = {}

    for idx, row in df.iterrows():
        slot_name = row['Title']  # slot 단위: Title 컬럼
        texts = [str(row[col]) for col in ['Product_Description','Shipping','Pricing','Shopping_Guarantee'] if pd.notna(row[col])]

        # 이미지 처리
        image_paths = str(row['Images']).split(",") if pd.notna(row['Images']) else []
        images = []
        for img_path in image_paths:
            img_path = img_path.strip()
            ocr_texts = run_ocr(img_path)
            role = classify_image_role(ocr_texts)
            images.append({
                "local_path": img_path,
                "ocr_text": " ".join(ocr_texts),
                "role": role
            })

        # TEXT explicitness
        text_out = judge_text_explicitness(slot_name, texts, product_name)
        # IMAGE explicitness
        image_out = judge_image_explicitness(slot_name, images, product_name)

        # Merge explicitness
        explicit_raw = merge_explicitness(text_out['text_explicitness'], image_out['image_explicitness'])

        # 결과 저장
        data[slot_name] = {
            "text_evidence": texts,
            "image_evidence": images,
            "text_explicitness": text_out['text_explicitness'],
            "image_explicitness": image_out['image_explicitness'],
            "explicitness_raw": explicit_raw,
            "anticipated_questions": text_out['anticipated_questions'] + image_out['anticipated_questions']
        }

    Path("vob_result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("✅ vob_result.json 생성 완료")

# --- 실행 예시 ---
csv_path = "/content/drive/MyDrive/innisfree-vob-voc-agent/data/seeds/VOB_serum.csv"
run_pipeline_from_csv_with_images(csv_path, "Innisfree Green Tea Seed Serum")

# ------------------------------
# 실행
# ------------------------------
if __name__ == "__main__":
    run_pipeline_from_csv(
        csv_path,
        product_name="Innisfree Green Tea Seed Serum"
    )