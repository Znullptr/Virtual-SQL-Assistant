# 🧠 AI SQL Chatbot

An intelligent Voice&Text chatbot built with FastAPI and React, converts natural language questions into SQL queries using OpenAI's GPT models. It executes queries on your database, returns structured results, and supports chart and PDF generation.


## 🌟 Core Functionalities

### 1. 🧾 Smart Question Answering
> Get instant answers to general knowledge or business-specific questions using LLMs.

![General Question Answering](images/general_qa.png)

---

### 2. 🗄️ Database Query Assistant
> Convert natural language to SQL, execute it on your DB, and/or receive structured responses or downloadable results.

![Database Queries](images/db_queries1.png)
![Database Queries](images/db_queries2.png)

---

### 3. 🎙️ Voice-to-Text Transcription
> Transcribe voice recordings into text using OpenAI's Whisper for hands-free chatbot interaction.

![Voice Transcription](images/voice_to_text.png)

---

### 4. 📄 PDF Report Generator
> Generate styled PDF reports for orders or custom data using Jinja2 templates and xhtml2pdf.

![PDF Generation](images/pdf_report.png)

---

### 5. 🔁 Custom Model Retraining
> Upload example question-answer pairs to improve the chatbot's SQL understanding on your specific schema.

![Retraining Model](images/retrain_model.png)
## 🛠️ Tech Stack

- **FastAPI** & **Pydantic**
- **OpenAI** + **LangChain**
- **Whisper** (Speech-to-text)
- **pandas**, **matplotlib**, **xhtml2pdf**
- **Jinja2** (for PDF templating)
- **React** (for frontend developement)

---

## 🚀 Features

- 🎯 Natural language → SQL query conversion
- 📊 Chart generation based on query results (bar, line, pie, scatter)
- 📄 PDF report generation for specific queries
- 📥 Export SQL results to Excel
- 🗣️ Whisper-based voice transcription
- 🔁 Retrain chatbot with custom examples
- 🔌 Built-in streaming response support (OpenAI API)
---

1. **Interface graphique** :

