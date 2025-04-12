import time
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from Utils.pdfReader import extract_text_from_pdf, extract_text_from_pdf_new,extract_text_and_metadata_new,text_to_lines
from Utils.helpers import normalize_text
import uuid
import os
import random
from Strategies.chunkers import semantic_chunking_mark_down,custom_semantic_chunking,recursive_chunking_method, chunk_text_with_overlap_object_return,sentence_chunking,split_sentences_by_spacy,chunk_markdown_by_header
from Utils.markdown import convert_to_markdown_return_string
from dotenv import load_dotenv
import nest_asyncio
import asyncio
nest_asyncio.apply()

load_dotenv()

app = FastAPI()
# Allow requests from your Angular app (localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # 👈 Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/chunk/text-overlap-file-upload")
async def chunk_text_overlap_upload(file: UploadFile = File(...), overlap: int = 1):
    """
    Processes the uploaded PDF file, extracts text, applies chunking, and measures execution time.

    :param file: The uploaded PDF file.
    :param overlap: Number of overlapping sentences per chunk (default = 1).
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Generate a unique filename using UUID and timestamp (milliseconds)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

        # Save the uploaded file to the 'Documents/' directory with the unique filename
        file_path = os.path.join("Documents", unique_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(file_path)

        # Apply chunking
        chunks, chunking_time = chunk_text_with_overlap_object_return(normalize_text(text), overlap=overlap)

        # Calculate total execution time (includes PDF extraction and chunking)
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": random.sample(chunks, min(5, len(chunks)))  # Select up to 5 random chunks
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/chunk/text-overlap")
def chunk_text_overlap(overlap: int = 1):
    """
    Processes the PDF from a given file path, extracts text, applies chunking, and measures execution time.

    :param pdf_path: Path to the PDF file.
    :param overlap: Number of overlapping sentences per chunk (default = 1).
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Extract text from PDF
        text = extract_text_from_pdf('Documents/CH3-data.pdf')

        # Apply chunking
        chunks, chunking_time = chunk_text_with_overlap_object_return(normalize_text(text), overlap=overlap)

        # Calculate total execution time (includes PDF extraction and chunking)
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks  # Return only first 5 chunks for preview
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/chunk/recursive-chunking-upload")
async def recursive_chunk_upload(file: UploadFile = File(...)):
    """
    Processes the uploaded PDF file, extracts text, applies chunking, and measures execution time.

    :param file: The uploaded PDF file.
    :param overlap: Number of overlapping sentences per chunk (default = 1).
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Generate a unique filename using UUID and timestamp (milliseconds)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

        # Save the uploaded file to the 'Documents/' directory with the unique filename
        file_path = os.path.join("Documents", unique_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(file_path)
        chunk_size: int = 500
        chunk_overlap: int = 50
        # Apply chunking
        chunks = recursive_chunking_method(normalize_text(text), chunk_size,chunk_overlap)

        # Calculate total execution time (includes PDF extraction and chunking)
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            # "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": random.sample(chunks, min(5, len(chunks)))  # Select up to 5 random chunks
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/chunk/recursive-chunking")
def recursive_chunk(chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Processes the PDF from a given file path, extracts text, applies chunking, and measures execution time.

    :param pdf_path: Path to the PDF file.
    :param overlap: Number of overlapping sentences per chunk (default = 1).
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Extract text from PDF
        text = extract_text_from_pdf('Documents/CH3-data.pdf')

        # Apply chunking
        chunks = recursive_chunking_method(normalize_text(text), chunk_size,chunk_overlap)
        # Calculate total execution time (includes PDF extraction and chunking)
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            # "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks[:5]  # Return only first 5 chunks for preview
        }

    except Exception as e:
        return {"error": str(e)}
    

@app.post("/chunk/sentence-chunking")
def sentence_chunk(
    text: str = Body(
        "Today was a fun day. I had lots of ice cream. I also met my best friend Sally and we played together at the new playground.",
        embed=True)):
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Apply chunking
        chunks = sentence_chunking(text)
        # Calculate total execution time
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/chunk/sentence-chunking-spacy")
def sentence_chunk():
    try:
        # Start total execution timer
        start_total_time = time.time()

        text = extract_text_from_pdf('Documents/CH3-data.pdf')
        # Apply chunking
        chunks = split_sentences_by_spacy(normalize_text(text), 20)
        # Calculate total execution time
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/text")
def sentence_chunk():
    try:
        # Start total execution timer
        text = extract_text_and_metadata_new('Documents/CH3-data.pdf')

        return text
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/chunking-by-headers")
def chunking_by_headers():
    try:        
        markdownstring = convert_to_markdown_return_string('Documents/CH3-data.pdf')
        chunks = chunk_markdown_by_header(markdownstring)
        return {
            "total_chunks": len(chunks),
            # "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks
        }
    except Exception as e:
            return {"error": str(e)}
    

@app.get("/chunking-by-sematics")
def chunking_by_headers():
    try:        
        document_lines = [
            "Artificial intelligence is transforming industries.",
            "Machine learning, a subset of AI, enables systems to learn from data.",
            "Deep learning techniques involve neural networks.",
            "Blockchain technology secures decentralized transactions.",
            "Smart contracts are self-executing agreements stored on the blockchain.",
            "Natural language processing helps machines understand human language.",
            "Large language models like GPT-4 generate human-like text."
        ]

        text = extract_text_from_pdf_new('Documents/CH3-data.pdf')
        lines = text_to_lines(text)
        print(lines)
        print(len(lines))
        chunks = custom_semantic_chunking(document_lines)
        return {
            "total_chunks": len(chunks),
            # "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks
        }
    except Exception as e:
            return {"error": str(e)}
    

@app.post("/chunk/chunking-by-sematics-upload")
async def chunking_by_sematics_upload(file: UploadFile = File(...)):
    """
    Processes the uploaded PDF file, extracts text, applies chunking, and measures execution time.

    :param file: The uploaded PDF file.
    :param overlap: Number of overlapping sentences per chunk (default = 1).
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Generate a unique filename using UUID and timestamp (milliseconds)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

        # Save the uploaded file to the 'Documents/' directory with the unique filename
        file_path = os.path.join("Documents", unique_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract and clean up text
        raw_text = extract_text_from_pdf(file_path)
        normalized_text = normalize_text(raw_text)

        # Split text into sentences or lines before chunking
        sentences = [s.strip() for s in normalized_text.split('.') if s.strip()]  # Simple sentence splitting
        if len(sentences) < 2:
            raise ValueError("Not enough text content for semantic chunking.")
        
        # Call semantic chunking
        semantic_chunks = custom_semantic_chunking(sentences)

        flattened_chunks = [" ".join(chunk) for chunk in semantic_chunks]

        total_execution_time = time.time() - start_total_time


        return {
            "total_chunks": len(flattened_chunks),
            # "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": random.sample(flattened_chunks, min(5, len(flattened_chunks)))  # Select up to 5 random chunks
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/chunking-by-sematics-new")
def chunking_by_headers_new():
    try:        
        document_lines = [
            "Artificial intelligence is transforming industries.",
            "Machine learning, a subset of AI, enables systems to learn from data.",
            "Deep learning techniques involve neural networks.",
            "Blockchain technology secures decentralized transactions.",
            "Smart contracts are self-executing agreements stored on the blockchain.",
            "Natural language processing helps machines understand human language.",
            "Large language models like GPT-4 generate human-like text."
        ]

        markdownstring = convert_to_markdown_return_string('Documents/CH3-data.pdf')
        chunks = semantic_chunking_mark_down(markdownstring)
        return {
            "total_chunks": len(chunks),
            # "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": chunks
        }
    except Exception as e:
            return {"error": str(e)}


@app.post("/chunk/masc-chunking-upload")
def chunking_by_headers_new_upload(file: UploadFile = File(...)):
    """
    Processes the uploaded PDF file, extracts text, applies chunking, and measures execution time.

    :param file: The uploaded PDF file.
    :return: JSON response with chunks and execution time.
    """
    try:
        # Start total execution timer
        start_total_time = time.time()

        # Generate a unique filename using UUID and timestamp (milliseconds)
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

        # Save the uploaded file to the 'Documents/' directory with the unique filename
        file_path = os.path.join("Documents", unique_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())  # Sync version, no await


        # Extract text from the uploaded PDF
        markdownstring = convert_to_markdown_return_string(file_path)
        chunks = semantic_chunking_mark_down(markdownstring)

        # Calculate total execution time (includes PDF extraction and chunking)
        total_execution_time = time.time() - start_total_time

        return {
            "total_chunks": len(chunks),
            # "chunking_time": f"{chunking_time:.4f} seconds",
            "total_execution_time": f"{total_execution_time:.4f} seconds",
            "chunks": random.sample(chunks, min(5, len(chunks)))  # Select up to 5 random chunks
        }

    except Exception as e:
        return {"error": str(e)}