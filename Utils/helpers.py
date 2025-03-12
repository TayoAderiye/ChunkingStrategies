
import re


def normalize_text(s, sep_token=" \n "):
    """Perform text cleaning by removing redundant whitespace and cleaning punctuation."""
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.replace("#", "")
    s = s.strip()    
    return s