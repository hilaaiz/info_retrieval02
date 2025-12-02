import os
import shutil

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


