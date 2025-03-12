import pdfplumber
from markdownify import markdownify
from collections import defaultdict
import re
from llama_parse import LlamaParse
import os
import re



def convert_to_markdown(pdf_path, output_file_path):
    parser = LlamaParse(api_key=os.getenv("LLAMA_API_KEY"), result_type='markdown')
    documents = parser.load_data(pdf_path) 
    # Open the output file for writing
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for doc in documents:
            # Extract the text content from the Document object
            text = doc.text  

            # Split text into lines to process headers
            lines = text.split('\n')

            # Flag to track if the previous NON-BLANK line was a header
            prev_header_line_index = None  

            for i, line in enumerate(lines):
                stripped_line = line.strip()  # Remove surrounding spaces
                
                if stripped_line.startswith('#'):  # Check if the line is a header
                    header_content = stripped_line.lstrip('#').strip()

                    # Check if the header contains ONLY a number (e.g., "2.1", "1.10") or a Roman numeral (e.g., "(ii)", "(IV)")
                    if re.match(r"^\d+(\.\d+)*$", header_content) or re.match(r"^\([ivxlcdm]+\)$", header_content, re.IGNORECASE):  
                        lines[i] = '# ' + header_content  # Convert to # subheading
                    elif prev_header_line_index is not None:
                        lines[i] = '# ' + stripped_line.lstrip('#').strip()  # Keep it as #
                    else:
                        lines[i] = '## ' + stripped_line.lstrip('#').strip()  # First header becomes ##

                    prev_header_line_index = i  # Update index of last header line
                elif stripped_line:  # Reset only if it's a non-blank, non-header line
                    prev_header_line_index = None  

            # Write processed content to file
            output_file.write("\n".join(lines) + "\n\n")  

def convert_to_markdown(pdf_path, output_file_path):
    parser = LlamaParse(api_key=os.getenv("LLAMA_API_KEY"), result_type='markdown')
    documents = parser.load_data(pdf_path) 
    # Open the output file for writing
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for doc in documents:
            # Extract the text content from the Document object
            text = doc.text  

            # Split text into lines to process headers
            lines = text.split('\n')

            # Flag to track if the previous NON-BLANK line was a header
            prev_header_line_index = None  

            for i, line in enumerate(lines):
                stripped_line = line.strip()  # Remove surrounding spaces
                
                if stripped_line.startswith('#'):  # Check if the line is a header
                    header_content = stripped_line.lstrip('#').strip()

                    # Check if the header contains ONLY a number (e.g., "2.1", "1.10") or a Roman numeral (e.g., "(ii)", "(IV)")
                    if re.match(r"^\d+(\.\d+)*$", header_content) or re.match(r"^\([ivxlcdm]+\)$", header_content, re.IGNORECASE):  
                        lines[i] = '# ' + header_content  # Convert to # subheading
                    elif prev_header_line_index is not None:
                        lines[i] = '# ' + stripped_line.lstrip('#').strip()  # Keep it as #
                    else:
                        lines[i] = '## ' + stripped_line.lstrip('#').strip()  # First header becomes ##

                    prev_header_line_index = i  # Update index of last header line
                elif stripped_line:  # Reset only if it's a non-blank, non-header line
                    prev_header_line_index = None  

            # Write processed content to file
            output_file.write("\n".join(lines) + "\n\n")  


def convert_to_markdown_return_string(pdf_path)-> str:
    parser = LlamaParse(api_key=os.getenv("LLAMA_API_KEY"), result_type='markdown')
    documents = parser.load_data(pdf_path)
    
    # Use a list to accumulate the markdown content instead of writing to a file
    markdown_content = []
    
    for doc in documents:
        # Extract the text content from the Document object
        text = doc.text  

        # Split text into lines to process headers
        lines = text.split('\n')

        # Flag to track if the previous NON-BLANK line was a header
        prev_header_line_index = None  

        for i, line in enumerate(lines):
            stripped_line = line.strip()  # Remove surrounding spaces
            
            if stripped_line.startswith('#'):  # Check if the line is a header
                header_content = stripped_line.lstrip('#').strip()

                # Check if the header contains ONLY a number (e.g., "2.1", "1.10") or a Roman numeral (e.g., "(ii)", "(IV)")
                if re.match(r"^\d+(\.\d+)*$", header_content) or re.match(r"^\([ivxlcdm]+\)$", header_content, re.IGNORECASE):  
                    lines[i] = '# ' + header_content  # Convert to # subheading
                elif prev_header_line_index is not None:
                    lines[i] = '# ' + stripped_line.lstrip('#').strip()  # Keep it as #
                else:
                    lines[i] = '## ' + stripped_line.lstrip('#').strip()  # First header becomes ##
                    
                prev_header_line_index = i  # Update index of last header line
            elif stripped_line:  # Reset only if it's a non-blank, non-header line
                prev_header_line_index = None  

        # Append the processed content to the markdown_content list
        markdown_content.append("\n".join(lines))
    
    # Return the final markdown as a string
    return "\n\n".join(markdown_content)