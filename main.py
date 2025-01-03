import os
import json
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import LangChain components
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time

# Load API key for ChatGroq
GROQ_API_KEY = "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"  # Replace with your actual API key

# Initialize ChatGroq model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# Define a prompt template for context-aware responses
prompt_template = ChatPromptTemplate.from_template(
    """
    Use the provided context and conversation history to answer the user's question.
    Please provide the most accurate response based on the question.
    Provide only the response, not any extra stuff, like 'based on the context and history'.

    <context>
    {context}
    </context>

    <conversation_history>
    {history}
    </conversation_history>

    Question: {input}
    """
)

def process_documents_and_answer_with_context(pdf_path, question, history, chunk_size=1000, chunk_overlap=200):
    """
    Process the given PDF document, generate embeddings, and answer the question with context awareness.

    Args:
        pdf_path (str): Path to the PDF file.
        question (str): User's question to be answered.
        history (list): Conversation history [(user_question, model_answer), ...].
        chunk_size (int): Size of document chunks for splitting.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        dict: Contains the answer, response time, and updated history.
    """
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_documents = text_splitter.split_documents(docs)

    # Generate embeddings for the chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Create a retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Format the conversation history for the prompt
    formatted_history = "\n".join(f"User: {q}\nModel: {a}" for q, a in history)
    context = "\n".join([doc.page_content for doc in final_documents])

    # Generate a response from the retrieval chain
    start_time = time.process_time()
    response = retrieval_chain.invoke({
        'input': question,
        'context': context,
        'history': formatted_history
    })
    response_time = time.process_time() - start_time

    # Extract the answer and update the history
    answer = response.get('answer', 'No answer found.')
    history.append((question, answer))

    return {
        "answer": answer,
        "response_time": response_time,
        "history": history
    }

# Directories for uploaded files and session histories
UPLOAD_DIR = "uploaded_files"
HISTORY_DIR = "session_histories"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
origins = ["*"]  # Update to restrict domains in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_history(session_id: str):
    """Load session history from a JSON file."""
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

def save_history(session_id: str, history: list):
    """Save session history to a JSON file."""
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(history_file, "w") as file:
        json.dump(history, file, indent=4)

@app.post("/ask/")
async def ask_question(
    question: str = Form(...),
    filename: str = Form(...),
    session_id: str = Form(...)  # Unique session identifier
):
    """
    Endpoint to answer a question using the uploaded PDF and session history.
    """
    pdf_path = os.path.join(UPLOAD_DIR, session_id + "__" + filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found.")

    # Load session history
    history = load_history(session_id)

    # Process documents and get the response
    result = process_documents_and_answer_with_context(pdf_path, question, history)

    # Save updated session history
    save_history(session_id, result["history"])

    return JSONResponse(content={
        "answer": result["answer"],
        "response_time": result["response_time"],
        "history": result["history"]
    })

@app.post("/upload/")
async def upload_pdf(file: UploadFile, session_id: str = Form(...)):
    """
    Endpoint to upload a PDF file.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the file locally
    file_path = os.path.join(UPLOAD_DIR, session_id + "__" + file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"message": "PDF uploaded successfully.", "filename": file.filename}

if __name__ == "__main__":
    # Start the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
