Frontend Repository: [https://github.com/jainbhavit2018/aiplanet-frontend](https://github.com/jainbhavit2018/aiplanet-frontend)\
Deployment Link: [https://pdf-chatbot-assignment.vercel.app/](https://pdf-chatbot-assignment.vercel.app/)

# AI Planet Backend

A FastAPI-based backend service for processing PDFs and answering questions using LangChain and Groq LLM.

## Overview

This application provides a REST API that allows users to:
- Upload PDF documents
- Ask questions about the uploaded documents
- Get AI-generated answers based on the document content
- Maintain conversation history across sessions

## Architecture

The application uses:
- **FastAPI** for the REST API framework
- **LangChain** for document processing and LLM interactions
- **Groq** as the Large Language Model
- **FAISS** for vector storage and similarity search
- **HuggingFace** embeddings for text processing
- **PyPDF** for PDF document handling

The workflow:
1. Users upload PDF documents which are stored locally
2. Documents are split into chunks and converted to embeddings
3. Questions are processed against the document context using LangChain
4. Responses are generated using the Groq LLM
5. Conversation history is maintained per session

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jainbhavit2018/aiplanet-backend.git
cd aiplanet-backend
```

2. Create a virtual environment (Windows):
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# To deactivate when done
deactivate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Update the Groq API key:
Replace `GROQ_API_KEY` in `main.py` with your actual Groq API key.

6. Run the application:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Documentation

### Upload PDF
```
POST /upload/
Content-Type: multipart/form-data

Parameters:
- file: PDF file (required)
- session_id: String (required)

Response:
{
    "message": "PDF uploaded successfully.",
    "filename": "document.pdf"
}
```

### Ask Question
```
POST /ask/
Content-Type: multipart/form-data

Parameters:
- question: String (required)
- filename: String (required)
- session_id: String (required)

Response:
{
    "answer": "AI-generated answer",
    "response_time": float,
    "history": [
        ["previous question", "previous answer"],
        ["current question", "current answer"]
    ]
}
```

## Project Structure
```
aiplanet-backend/
├── main.py              # Main application file
├── requirements.txt     # Project dependencies
├── uploaded_files/      # Directory for stored PDFs
└── session_histories/   # Directory for conversation histories
```

## Error Handling

The API includes error handling for:
- Invalid file types (non-PDF files)
- Missing files
- Missing required parameters
- Server-side processing errors

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
