import os
import shutil

import re

import html


# תיקיית הסקריפט (scripts/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# עולים תיקייה אחת למעלה - INFO_RETRIEVAL02
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# תיקיות המקור (UK/US) שנמצאות תחת INFO_RETRIEVAL02 ולא תחת scripts
uk_folder = os.path.join(ROOT_DIR, "UK_british_debates_text_files_normalize")
us_folder = os.path.join(ROOT_DIR, "US_congressional_speeches_Text_Files")

# תיקיית output (allData) גם ברמת ROOT
output_folder = os.path.join(ROOT_DIR, "allData")

# יצירת התיקייה המאוחדת
os.makedirs(output_folder, exist_ok=True)

def clean_congressional_text(raw_text):

    """

    Cleans US Congressional Record file.

    Keeps only the speech text inside <pre> ... </pre>.

    Removes HTML, entities, and normalizes punctuation spacing.

    """



    # --- 1. Extract only the <pre>...</pre> content ---

    pre_match = re.search(r"<pre>(.*?)</pre>", raw_text, flags=re.DOTALL | re.IGNORECASE)

    if pre_match:

        text = pre_match.group(1)

    else:

        # If no <pre> block exists, fallback to full text

        text = raw_text



    # --- 2. Remove all HTML tags (<a>, <br>, etc.) ---

    text = re.sub(r"<.*?>", " ", text)



    # --- 3. Convert HTML entities: &#x27; → ', &amp; → &, etc. ---

    text = html.unescape(text)



    # --- 4. Separate punctuation from words ---

    # Add spaces around ANY non-alphanumeric char

    text = re.sub(r'([^A-Za-z0-9])', r' \1 ', text)



    # --- 5. Normalize multiple spaces ---

    text = re.sub(r'\s+', ' ', text)



    # Do NOT strip — to stay consistent with your earlier cleaning rules

    return text

def copy_and_rename(src_folder, prefix):
    print(f"Scanning: {src_folder}")
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        if not os.path.isfile(src_path):
            continue

        new_name = f"{prefix}_{filename}"
        dst_path = os.path.join(output_folder, new_name)

        shutil.copy(src_path, dst_path)
        print("Copied:", new_name)

copy_and_rename(uk_folder, "UK")
copy_and_rename(us_folder, "US")

print("✓ המיזוג הסתיים! כל הקבצים נמצאים בתיקיית allData.")


