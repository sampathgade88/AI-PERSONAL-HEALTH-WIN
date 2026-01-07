 ## 📌 *Hackathon Project: GenAI Multi-Agent Healthcare System*

# 🧠🚑 HealthGuard Agents – Multi-Agent AI Healthcare Companion

This repository contains a *multi-agent AI healthcare system* developed for the *GenAI Hackathon*, designed to monitor health, predict disease risks, and provide personalized wellness recommendations using autonomous AI agents.

---

## 🎯 *Objective*

To develop an intelligent healthcare companion using *AI agents, machine learning, and user health data* in order to:

* Predict early-stage disease risks
* Understand user symptoms through triage-style questioning
* Monitor lifestyle patterns like sleep, stress, diet, and activity
* Generate personalized care plans
* Offer daily health coaching
* Send real-time alerts when risk factors increase

---

## 🧰 *Tools and Technologies Used*

* Python
* Node.js / FastAPI
* LangChain / AutoGen (Multi-Agent Framework)
* OpenAI / Llama Models
* Scikit-learn
* Firebase / MongoDB
* Flutter / React Native

---

## 🩺 *System Overview*

HealthGuard Agents is powered by *five specialized AI agents* that work together like a virtual medical team:

* *Lifestyle & Behavior Agent* – tracks habits and detects irregular patterns
* *Symptom & Triage Agent* – analyzes symptoms and asks medical questions
* *Predictive Health Agent* – computes disease risk scores
* *Medical Knowledge Agent* – explains health conditions and precautions
* *Care Plan Agent* – generates diet, exercise, and stress-relief plans

---

## 📦 *Dataset / Models Used*

This project uses ML models trained on:

* PIMA Diabetes Dataset
* UCI Heart Disease Dataset
* NHANES Lifestyle Dataset

Models used include:

* Logistic Regression
* Random Forest
* LightGBM
* LLM-based reasoning models

---

## 📊 *What Was Done*

1. *Designed and implemented a multi-agent system*

   * Created agents with unique healthcare tasks
   * Enabled agent-to-agent communication

2. *Built risk prediction models*

   * Diabetes risk
   * Hypertension risk
   * Stress/fatigue risk

3. *Developed symptom triage flow*

   * Diagnostic questioning
   * Severity estimation

4. *Added lifestyle monitoring logic*

   * Sleep tracking
   * Stress/diet/activity pattern analysis

5. *Generated personalized care plans*

   * Weekly routines
   * Diet and exercise recommendations

6. *Created real-time alerts*

   * Notifies users when risk scores increase

---

## 🔍 *Key Insights*

* 🧠 *Proactive Health Monitoring:*
  Agents help detect health issues before they become serious.

* 🔗 *Multi-Agent Collaboration:*
  Agents communicate like a real medical team for better accuracy.

* ❤️ *Personalized Care:*
  Recommendations change based on user behavior and symptoms.

* 📉 *Risk Reduction:*
  Early warnings improve lifestyle habits and reduce disease chances.

* 👨‍⚕️ *Accessible Healthcare:*
  Helpful for users with limited access to medical professionals.

---
# 🩺 MedQuAD Medical Q&A Chatbot

A *domain-specific chatbot* for answering medical and health-related questions using the *NIH MedQuAD dataset.The chatbot combines **semantic search* and a *contextual question-answering model* to provide accurate, refined answers.

---

## 🌟 Features

* *Domain-specific Q&A*: Focused on medical questions.
* *Semantic Search: Retrieves the most relevant Q&A entries using **sentence embeddings* and *FAISS similarity search*.
* *Contextual Answer Refinement: Enhances the retrieved answer using a **RoBERTa-based QA model*.
* *Interactive Interface: Built with **Gradio*, providing a user-friendly chat interface.
* *Lightweight & Efficient*: Uses all-MiniLM-L6-v2 for embeddings and deepset/roberta-base-squad2 for QA.

---

## 🛠️ Tech Stack

* *Python*
* *Pandas & NumPy* – data manipulation
* *Sentence-Transformers* – semantic embeddings
* *FAISS* – similarity search
* *Transformers (Hugging Face)* – question-answering pipeline
* *Gradio* – web-based interactive interface

---

## 📂 Dataset

* *NIH MedQuAD*: A curated dataset of medical question-answer pairs.
* Required columns: Question, Answer

---

## ⚡ How It Works

1. Load and preprocess the *MedQuAD dataset*.
2. Encode all questions into *sentence embeddings*.
3. Build a *FAISS index* for fast semantic similarity search.
4. For a user query:

   * Find the most similar question(s) from the dataset.
   * Refine the retrieved answer using a *RoBERTa QA model*.
5. Return the *original answer, **refined answer, and **source*.

---

## 💬 Example

*Query:* "What are the early signs of diabetes?"
*Answer:* "Refined answer based on MedQuAD dataset context."
*Source:* "NIH MedQuAD"
