# app.py - MedQuAD Medical Q&A Chatbot (PyCharm Version) with Generative AI

import os
import logging
import pandas as pd
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------
# FIX 1: Disable HuggingFace symlink warnings (Windows)
# -------------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------------------------------------------------------
# FIX 2: Reduce Gradio log noise
# -------------------------------------------------------
logging.getLogger("gradio").setLevel(logging.ERROR)

# -------------------------------------------------------
# STEP 1: Load MedQuAD Dataset
# -------------------------------------------------------
def load_dataset():
    try:
        df = pd.read_csv("medquad.csv")
    except FileNotFoundError:
        print("❌ ERROR: 'medquad.csv' not found. Place the file in the project folder.")
        raise SystemExit

    df = df.rename(columns={"Question": "question", "Answer": "answer"})
    df["source"] = "NIH MedQuAD"

    if "question" not in df.columns or "answer" not in df.columns:
        print("❌ ERROR: Dataset must contain 'Question' and 'Answer' columns.")
        raise SystemExit

    df = df[["question", "answer", "source"]]
    print("✅ Dataset loaded successfully!")
    return df

df = load_dataset()

# -------------------------------------------------------
# STEP 2: Build Embeddings + FAISS Index
# -------------------------------------------------------
print("🔄 Creating sentence embeddings...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
questions = df["question"].astype(str).tolist()

question_embeddings = embedder.encode(questions, show_progress_bar=True)
question_embeddings = np.array(question_embeddings).astype("float32")

index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

print(f"✅ FAISS index created with {index.ntotal} entries.")

# -------------------------------------------------------
# STEP 3: Semantic Search
# -------------------------------------------------------
def semantic_search(query, top_k=3):
    query_emb = embedder.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "question": df.iloc[idx]["question"],
            "answer": df.iloc[idx]["answer"],
            "source": df.iloc[idx]["source"],
            "distance": float(distances[0][i])
        })
    return results

# -------------------------------------------------------
# STEP 4: Load QA Refinement Model
# -------------------------------------------------------
print("🤖 Loading QA refinement model...")
qa_model = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# -------------------------------------------------------
# STEP 5: Load Generative AI Model (Optional: GPT-Neo)
# -------------------------------------------------------
print("🤖 Loading generative AI model...")
gen_model_name = "EleutherAI/gpt-neo-125M"  # lightweight for demo
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

def generate_answer(prompt, max_length=150):
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    outputs = gen_model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# -------------------------------------------------------
# STEP 6: Answer Query (Semantic + QA + Gen AI)
# -------------------------------------------------------
def answer_query(query):
    # Step 1: Semantic search
    results = semantic_search(query, top_k=1)
    if not results:
        return {
            "original_answer": "No matching information found.",
            "refined_answer": "No refined answer available.",
            "source": ""
        }

    best = results[0]
    context = str(best["answer"])

    if not context.strip():
        return {
            "original_answer": best["answer"],
            "refined_answer": "No refinement possible (empty context).",
            "source": best["source"]
        }

    # Step 2: QA refinement
    try:
        refined = qa_model(question=query, context=context)
        refined_answer = refined.get("answer", "Error: No answer extracted.")
    except Exception as e:
        print(f"⚠ QA refinement error: {e}")
        refined_answer = "Error refining answer."

    # Step 3: Generative AI expansion
    try:
        prompt = f"User asked: {query}\nContext: {context}\nProvide a clear and informative medical answer:"
        gen_answer = generate_answer(prompt, max_length=200)
    except Exception as e:
        print(f"⚠ Generative AI error: {e}")
        gen_answer = refined_answer  # fallback

    return {
        "original_answer": best["answer"],
        "refined_answer": gen_answer,
        "source": best["source"]
    }

# -------------------------------------------------------
# STEP 7: Gradio UI
# -------------------------------------------------------
def chat_interface(query):
    if not query.strip():
        return "❗ Please enter a question."
    result = answer_query(query)
    return f"""
### 🩺 Refined Answer
{result['refined_answer']}

**Source:** {result['source']}
"""

app = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(label="Ask a Medical Question"),
    outputs="markdown",
    title="🩺 MedQuAD Medical Q&A Chatbot + Generative AI",
    description="Ask any medical question. Uses semantic search + RoBERTa QA + GPT-Neo for expanded answers.",
)

# -------------------------------------------------------
# STEP 8: Launch App
# -------------------------------------------------------
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
