import PyPDF3
import re
import pdfplumber
from collections import defaultdict

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF3.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    document = open(pdf_path, 'rb')
    doc_helper = PyPDF3.PdfFileReader(document)

    finaltext = ''
    totalpages = doc_helper.getNumPages()
    
    for eachpage in range(totalpages):
        p = doc_helper.getPage(eachpage)
        indpagetext = p.extractText()
        finaltext += indpagetext
    
    document.close()
    return finaltext




def extract_text_and_metadata(pdf_path):
    """
    Extracts text and metadata from a PDF file, then dynamically detects and chunks text based on headings.

    :param pdf_path: Path to the PDF file.
    :return: A dictionary with metadata and sectioned text.
    """
    document = open(pdf_path, 'rb')
    doc_helper = PyPDF3.PdfFileReader(document)

    # Extract metadata
    metadata = doc_helper.getDocumentInfo()
    meta_dict = {key: metadata[key] for key in metadata} if metadata else {}

    # Extract text
    final_text = ''
    total_pages = doc_helper.getNumPages()
    
    for each_page in range(total_pages):
        p = doc_helper.getPage(each_page)
        page_text = p.extractText()
        final_text += page_text + '\n'

    document.close()

    # Regex to detect potential section headings
    heading_pattern = re.compile(r'^(?:\d{1,2}\.\s*)?[A-Z][A-Za-z\s]{2,50}$')

    sections = {}
    current_section = "UNKNOWN"
    sections[current_section] = []

    for line in final_text.split("\n"):
        stripped_line = line.strip()

        # Check if the line looks like a heading
        if heading_pattern.match(stripped_line):
            current_section = stripped_line
            sections[current_section] = []
        sections[current_section].append(stripped_line)

    # Convert list of lines into strings
    for section in sections:
        sections[section] = "\n".join(sections[section])

    return {
        "metadata": meta_dict,
        "sections": sections
    }


import PyPDF3
import re

def extract_text_and_metadata2(pdf_path):
    """
    Extracts text and metadata from a PDF file, then dynamically detects and chunks text based on headings.

    :param pdf_path: Path to the PDF file.
    :return: A dictionary with metadata and an ordered list of (section_title, content).
    """
    document = open(pdf_path, 'rb')
    doc_helper = PyPDF3.PdfFileReader(document)

    # Extract metadata
    metadata = doc_helper.getDocumentInfo()
    meta_dict = {key: metadata[key] for key in metadata} if metadata else {}

    # Extract text
    final_text = ''
    total_pages = doc_helper.getNumPages()
    
    for each_page in range(total_pages):
        p = doc_helper.getPage(each_page)
        page_text = p.extractText()
        final_text += page_text + '\n'

    document.close()

    # Improved heading detection regex:
    heading_pattern = re.compile(r'^(?:\d{1,2}(\.\d{1,2})?\s*[-–.]?\s*)?[A-Z][A-Za-z\s\-:,]{2,80}$')

    chunks = []
    current_section = "UNKNOWN"
    current_content = []

    for line in final_text.split("\n"):
        stripped_line = line.strip()

        # Check if the line is a potential heading
        if heading_pattern.match(stripped_line):
            # Save previous section
            if current_content:
                chunks.append((current_section, "\n".join(current_content)))
                current_content = []
            # Set new section title
            current_section = stripped_line

        current_content.append(stripped_line)

    # Save last section
    if current_content:
        chunks.append((current_section, "\n".join(current_content)))

    return {
        "metadata": meta_dict,
        "chunks": chunks  # List of (heading, text) tuples
    }


def extract_text_and_metadata_new(pdf_path):
    """
    Extracts text and metadata from a PDF file, dynamically detecting headings using font size and boldness.
    Groups words into full lines to avoid splitting headings into individual words.

    :param pdf_path: Path to the PDF file.
    :return: A dictionary with metadata and an ordered list of (section_title, content).
    """
    chunks = []
    current_section = "UNKNOWN"
    current_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        metadata = pdf.metadata
        font_sizes = []
        lines_by_y = defaultdict(list)

        # Process each page
        for page in pdf.pages:
            for block in page.extract_words(extra_attrs=["fontname", "size", "y0"]) or []:
                text, font_size, font_name, y0 = block["text"].strip(), block["size"], block["fontname"], block["y0"]
                lines_by_y[y0].append((text, font_size, font_name))  # Group words by Y coordinate
                font_sizes.append(font_size)

        # Determine a heading size threshold
        if font_sizes:
            heading_size_threshold = max(font_sizes) * 0.8  # 80% of max font size (adjustable)

        # Process lines
        for y, words in sorted(lines_by_y.items(), key=lambda x: x[0]):  # Sort by Y position
            line_text = " ".join([word[0] for word in words])  # Join words into full line
            line_font_sizes = [word[1] for word in words]
            line_fonts = [word[2] for word in words]

            # Check if the line is a heading
            is_heading = max(line_font_sizes) >= heading_size_threshold or any("Bold" in f for f in line_fonts)

            if is_heading:
                # Save the previous section
                if current_content:
                    chunks.append((current_section, "\n".join(current_content)))
                    current_content = []

                # Set new section title
                current_section = line_text

            current_content.append(line_text)

        # Save last section
        if current_content:
            chunks.append((current_section, "\n".join(current_content)))

    return {
        "metadata": metadata,
        "chunks": chunks  # List of (heading, text) tuples
    }




def extract_text_from_pdf_new(pdf_path):
    """
    Extracts text from a PDF file using PyPDF3.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    document = open(pdf_path, 'rb')
    doc_helper = PyPDF3.PdfFileReader(document)

    finaltext = ''
    totalpages = doc_helper.getNumPages()
    
    for eachpage in range(totalpages):
        p = doc_helper.getPage(eachpage)
        indpagetext = p.extractText()
        finaltext += indpagetext + "\n"  # Ensure text is separated by newlines
    
    document.close()
    return finaltext

# Function to convert extracted text into lines
def text_to_lines(text):
    """
    Converts a block of text into individual lines.

    :param text: Extracted text as a string.
    :return: List of lines.
    """
    lines = text.split("\n")  # Split based on new lines
    lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
    return lines