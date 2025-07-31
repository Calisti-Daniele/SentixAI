import re

def clean_text(text):
    import re
    text = re.sub(r"\s+", " ", text).strip()  # Rimuove \n, \t, doppi spazi, ecc.
    return text
