# 🧠 AI SQL Chatbot

An intelligent Voice&Text chatbot built with FastAPI and React, converts natural language questions into SQL queries using OpenAI's GPT models. It executes queries on your database, returns structured results, and supports chart and PDF generation.


## 🌟 Core Functionalities

### 1. 🧾 Smart Question Answering
> Get instant answers to general knowledge or business-specific questions using LLMs.

![General Question Answering](Images/general_qa.png)

---

### 2. 🗄️ Database Query Assistant
> Convert natural language to SQL, execute it on your DB, and/or receive structured responses or downloadable results.

![Database Queries](Images/db_queries2.png)
![Database Queries](Images/db_queries1.png)

---

### 3. 🎙️ Voice-to-Text Transcription
> Transcribe voice recordings into text using OpenAI's Whisper for hands-free chatbot interaction.

![Voice Transcription](Images/voice_to_text1.png)
![Voice Transcription](Images/voice_to_text2.png)

---

### 4. 📄 PDF Report Generator
> Generate styled PDF reports for orders or custom data using Jinja2 templates and xhtml2pdf.

![PDF Generation](Images/pdf_report.png)

---

### 5. 🔁 Custom Model Retraining
> Upload example question-answer pairs to improve the chatbot's SQL understanding on your specific schema.

![Retraining Model](Images/retrain_model.png)
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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

