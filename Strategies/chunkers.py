from typing import List
import spacy
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken as tk
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from Utils.helpers import normalize_text
import os
import openai
from time import sleep
from openai.error import RateLimitError

def chunk_text_with_overlap(text, overlap=1):
    """
    Splits text into overlapping sentence-based chunks.

    :param text: Input text to be chunked.
    :param overlap: Number of overlapping sentences between chunks.
    :return: List of overlapping sentence chunks.
    """
    # Load English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Process text with spaCy
    doc = nlp(text)

    # Extract sentences
    sentences = list(doc.sents)

    # Start timer for chunking
    start_time = time.time()

    # Create chunks with overlap
    chunks = []
    for i in range(len(sentences) - overlap):
        chunk = sentences[i : i + overlap + 1]
        chunks.append(" ".join(sent.text for sent in chunk))  # Combine sentences into a string

    # End timer for chunking
    chunking_time = time.time() - start_time

    return chunks, chunking_time

def recursive_chunking_method(text, chunk_size=500, chunk_overlap=50):
    """
    Recursively chunks the text into smaller segments until they fit the chunk_size.
    
    :param text: The text to be chunked.
    :param chunk_size: The maximum size for each chunk.
    :param chunk_overlap: The number of characters to overlap between chunks.
    :return: A list of text chunks.
    """
    # Create a text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split the text using the splitter
    chunks = splitter.split_text(text)
    
    return chunks

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# def recursive_chunking_method(text, chunk_size=500, chunk_overlap=50):

#     # Create a text splitter
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
#     # Split the text using the splitter
#     chunks = splitter.split_text(text)
    
#     return chunks

def chunk_text_with_overlap_object_return(text, overlap=1):
    """
    Splits text into overlapping sentence-based chunks and stores them as objects.

    :param text: Input text to be chunked.
    :param overlap: Number of overlapping sentences between chunks.
    :return: List of dictionaries, where each dictionary represents a chunk.
    """
    # Load English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Process text with spaCy
    doc = nlp(text)

    # Extract sentences
    sentences = list(doc.sents)

    # Start timer for chunking
    start_time = time.time()

    # Create chunks with overlap
    chunks = []
    for i in range(len(sentences) - overlap):
        chunk_text = " ".join(sent.text for sent in sentences[i : i + overlap + 1])  # Combine sentences
        chunks.append({
            "chunk_id": i + 1,
            "content": chunk_text,
            "num_sentences": overlap + 1
        })

    # End timer for chunking
    chunking_time = time.time() - start_time

    return chunks, chunking_time


def fixed_window_splitter(text: str, chunk_size: int = 1000) -> List[str]:
    """Splits text at given chunk_size"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# def sentence_chunking(text: str) -> List[str]:
#     """Splits text into sentences using spaCy."""
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     return [sent.text for sent in doc.sents]

def sentence_chunking(text: str):
    """Splits text into sentences using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    for i, sent in enumerate(doc.sents):
        chunks.append({
            "chunk_id": i + 1,
            "content": sent.text,
            "num_sentences": 1
        })
    return chunks

def count_tokens(string: str, encoding_name="cl100k_base") -> int:
    # Get the encoding
    encoding = tk.get_encoding(encoding_name)
    
    # Encode the string
    encoded_string = encoding.encode(string, disallowed_special=())

    # Count the number of tokens
    num_tokens = len(encoded_string)
    return num_tokens

def split_sentences_by_spacy(text, max_tokens, overlap=0, model="en_core_web_sm"):

    nlp = spacy.load(model) 
    doc = nlp(text)           
    sentences = [sent.text for sent in doc.sents]

    tokens_lengths = [count_tokens(sent) for sent in sentences]  
    chunks = []
    start_idx = 0
    while start_idx < len(sentences):
        current_chunk = []
        current_token_count = 0
        for idx in range(start_idx, len(sentences)):
            if current_token_count + tokens_lengths[idx] > max_tokens:
                break
            current_chunk.append(sentences[idx])
            current_token_count += tokens_lengths[idx]
        chunks.append(" ".join(current_chunk))
        if overlap >= len(current_chunk):  
            start_idx += 1
        else:    
         start_idx += len(current_chunk) - overlap
    return chunks


# from langchain.text_splitter import SemanticChunker
# from langchain.embeddings.openai import OpenAIEmbeddings

# def semantic_chunking_method(text, chunk_size=500, chunk_overlap=50, model="text-embedding-ada-002"):
#     # Create an embedding model
#     embeddings = OpenAIEmbeddings(model=model)
    
#     # Create a semantic chunker
#     splitter = SemanticChunker(embedding_model=embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
#     # Split the text using the chunker
#     chunks = splitter.split_text(text)
    
#     return chunks


# def chunk_markdown_by_header(markdown_text: str):
#     headers_to_split_on = [
#             ("##", "Header 1"),
#         ]
#     markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
#     md_header_splits = markdown_splitter.split_text(markdown_text)
#     return md_header_splits

def chunk_markdown_by_header(markdown_text: str):
    headers_to_split_on = [
        ("##", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    result = []
    for split in md_header_splits:
        # Access the 'metadata' attribute of the Document object
        header_1 = split.metadata.get("Header 1", "")  # Safely access Header 1
        
        # Access the 'page_content' attribute of the Document object
        page_content = getattr(split, "page_content", "")  # Safe access to page_content
        
        result.append({
            # "Header": header_1,
            "page_content": normalize_text(page_content)
        })
    
    return result

def get_headers(markdown_text: str):
    headers_to_split_on = [
        ("##", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    result = []
    for split in md_header_splits:
        # Access the 'metadata' attribute of the Document object
        header = split.metadata.get("Header 1", "")  # Safely access Header 1
        
        result.append({
            "Header": header,
            # "page_content": normalize_text(page_content)
        })
    
    return result

def custom_semantic_chunking(lines):
    chunks = []
    current_chunk = [lines[0]]

    for i in range(1, len(lines)):
        if are_lines_related(lines[i - 1], lines[i]):
            current_chunk.append(lines[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [lines[i]]

    # Append last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks





# Function to ask GPT if two lines are related
# def are_lines_related(line1, line2):
#     openai.api_type = os.getenv("OPENAI_API_TYPE")
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     openai.api_base = os.getenv("OPENAI_API_BASE")
#     openai.api_version = os.getenv("OPENAI_API_VERSION")

#     prompt = f"""
#     Determine if the following two lines belong to the same topic. 
#     Respond with ONLY "YES" or "NO".

#     Line 1: {line1}
#     Line 2: {line2}

#     Answer:
#     """
    
#     response = openai.ChatCompletion.create(
#         engine=os.getenv("TEXT_DEPLOYMENT_ID"),
#         messages=[{"role": "system", "content": "You are an AI that determines if two lines are related."},
#                   {"role": "user", "content": prompt}],
#         max_tokens=2,
#         temperature=0
#     )
    
#     answer = response["choices"][0]["message"]["content"].strip().upper()
#     return answer == "YES"


def are_lines_related(line1, line2):
    """
    Determines if two lines are semantically related using GPT.
    Retries indefinitely in case of rate limits.
    
    :param line1: First line of text.
    :param line2: Second line of text.
    :return: True if related, False otherwise.
    """
    # Set up OpenAI API credentials
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    prompt = f"""
    Determine if the following two lines belong to the same topic. 
    Respond with ONLY "YES" or "NO".

    Line 1: {line1}
    Line 2: {line2}

    Answer:
    """

    messages = [
        {"role": "system", "content": "You are an AI that determines if two lines are related."},
        {"role": "user", "content": prompt}
    ]

    wait_time = 2  # Initial retry wait time

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=os.getenv("TEXT_DEPLOYMENT_ID"),
                messages=messages,
                max_tokens=2,
                temperature=0
            )

            answer = response["choices"][0]["message"]["content"].strip().upper()
            return answer == "YES"

        except RateLimitError:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s wait




def are_chunks_related(chunk1, chunk2):
    """
    Determines if two chunks are semantically related using GPT.
    Retries indefinitely if rate limit is hit.

    :param chunk1: First chunk of text.
    :param chunk2: Second chunk of text.
    :return: True if related, False otherwise.
    """
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    prompt = f"""
    Determine if the following two sections of text belong to the same topic.
    Respond with ONLY "YES" or "NO".

    Section 1: {chunk1}
    Section 2: {chunk2}

    Answer:
    """

    messages = [
        {"role": "system", "content": "You are an AI that determines if two sections are related."},
        {"role": "user", "content": prompt}
    ]

    wait_time = 2  # Start with a 2-second delay for retries

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=os.getenv("TEXT_DEPLOYMENT_ID"),
                messages=messages,
                max_tokens=2,
                temperature=0
            )

            answer = response["choices"][0]["message"]["content"].strip().upper()
            return answer == "YES"

        except RateLimitError:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff (max 60s)



def semantic_chunking_mark_down(markdown_text):
    """
    Splits markdown by headers and then merges related chunks using GPT.

    :param markdown_text: The input markdown text.
    :return: A list of semantically meaningful chunks.
    """
    initial_chunks = chunk_markdown_by_header(markdown_text)
    result_chunks = []
    current_chunk = [initial_chunks[0]["page_content"]]

    for i in range(1, len(initial_chunks)):
        chunk1 = " ".join(current_chunk)  # Merge current chunk into text
        chunk2 = initial_chunks[i]["page_content"]  # Next chunk text

        if are_chunks_related(chunk1, chunk2):
            current_chunk.append(chunk2)
        else:
            result_chunks.append(" ".join(current_chunk))  # Merge current chunk
            current_chunk = [chunk2]  # Start a new chunk

    # Append the last chunk
    if current_chunk:
        result_chunks.append(" ".join(current_chunk))

    return result_chunks
