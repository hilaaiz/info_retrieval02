import re
import os
import glob
from pathlib import Path

def clean_congressional_record(text):
    """
    מנקה תמלולי קונגרס ומשאיר רק את הדיבורים של חברי הקונגרס
    """
    # שלב 1: חילוץ כל התוכן מתוך תגיות <pre>
    pre_contents = re.findall(r'<pre>(.*?)</pre>', text, re.DOTALL)
    
    cleaned_speeches = []
    
    for content in pre_contents:
        # שלב 2: הסרת שורות עם סוגריים מרובעים [] (כולל כותרות ומיקומי עמוד)
        lines = content.split('\n')
        lines = [line for line in lines if not re.search(r'\[.*?\]', line)]
        
        # שלב 3: הסרת קווים מפרידים (______), שורות מקור וקישורים
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # דלג על קווים מפרידים
            if re.match(r'^[_=]{3,}$', stripped):
                continue
            # דלג על שורות מקור
            if 'Congressional Record Online' in line or 'Government Publishing Office' in line:
                continue
            # דלג על קישורים לאתר
            if 'www.gpo.gov' in line or '<a href=' in line:
                continue
            # דלג על שורות תאריך בלבד
            if re.match(r'^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\w+\s+\d+,\s+\d{4}\s*$', stripped):
                continue
            filtered_lines.append(line)
        
        lines = filtered_lines
        
        # שלב 4: הסרת כותרות באותיות גדולות (שאינן דיבורים)
        filtered_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            # בדיקה אם השורה היא באותיות גדולות
            letters_only = re.sub(r'[^A-Za-z]', '', current_line)
            is_uppercase = len(letters_only) > 5 and letters_only.isupper()
            
            if is_uppercase:
                # בדיקה אם יש עוד שורות רצופות באותיות גדולות
                consecutive_uppercase = 1
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    next_letters = re.sub(r'[^A-Za-z]', '', next_line)
                    if len(next_letters) > 5 and next_letters.isupper():
                        consecutive_uppercase += 1
                        j += 1
                    else:
                        break
                
                # אם יש 2+ שורות רצופות באותיות גדולות - זו כותרת
                if consecutive_uppercase >= 2 or (is_uppercase and (i == 0 or not lines[i-1].strip())):
                    i = j
                    continue
            
            filtered_lines.append(lines[i])
            i += 1
        
        lines = filtered_lines
        
        # שלב 5: הסרת שורות שיש לפניהן ואחריהן שורה ריקה (הסבר על מיקום, כותרת משנה וכד')
        filtered_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            if current_line:
                has_empty_before = (i == 0 or not lines[i-1].strip())
                has_empty_after = (i == len(lines)-1 or not lines[i+1].strip())
                
                # תנאי שמסנן שורות קצרות וכלליות המופרדות ברווח
                if has_empty_before and has_empty_after and len(current_line.split()) < 10 and not current_line.endswith('.'):
                    i += 1
                    continue
            
            filtered_lines.append(lines[i])
            i += 1
        
        # חיבור השורות בחזרה
        cleaned_text = '\n'.join(filtered_lines)

        # הסרת רצף הגיבריש של הגרש (&#x27;)
        cleaned_text = cleaned_text.replace("&#x27;", "")
        
        # =========================================================================
        # שלב 6: הסרת תחילת פסקאות - כל התבניות של מי מדבר
        # =========================================================================

        # תבנית 1: הסרת פתיח שמכיל Ms./Mrs./Mr. + שם + Mr. Speaker. 
        cleaned_text = re.sub(
            # מתחילת שורה (או אחרי שורה ריקה)
            r'(^|\n)\s*(Ms\.|Mrs\.|Mr\.)' 
            # לוכד את שם המשפחה 
            r'\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)*|[A-Z]+(\s+of\s+[A-Za-z]+)?)\.?' 
            # לוכד את הפנייה ליו"ר (Mr. Speaker) ואת סימני הפיסוק הנלווים
            r'(\s*Mr\.\s*Speaker[,.]?)?\s*',
            r'\1', # שומר רק את ה-\n או ה-^
            cleaned_text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # תבנית 2: הסרת אזכורים מנומסים של תארים/שמות (כדי לנקות פסקאות פתיחה/רקע קצרות)
        cleaned_text = re.sub(
            r'(^|\n)\s*(Dr\.|Deputy|Superintendent|His\s+valiant|Charles|Ms\.|Mrs\.|Mr\.)\s+[A-Z][a-z]+(\s+of\s+[A-Z][a-z]+)?\s*[^.?!]{10,100}(?=\s*[\.\?!])',
            r'\1',
            cleaned_text,
            flags=re.MULTILINE
        )
        
        # הסרת שורות ריקות מרובות
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        # הסרת קווי הפרדה מהתחילה והסוף
        cleaned_text = re.sub(r'^[=\s]+', '', cleaned_text)
        cleaned_text = re.sub(r'[=\s]+$', '', cleaned_text)
        
        # הסרת רווחים מיותרים
        cleaned_text = cleaned_text.strip()
        
        if cleaned_text:
            cleaned_speeches.append(cleaned_text)
    
    # הפרדה בין נאומים שונים
    return '\n'.join(cleaned_speeches)


def clean_file(input_file_path, output_file_path):
    """
    מנקה קובץ ושומר את התוצאה.
    """
    # קריאת הקובץ
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"שגיאה: הקובץ '{input_file_path}' לא נמצא.")
        return
    except Exception as e:
        print(f"שגיאה בקריאת הקובץ '{input_file_path}': {e}")
        return

    
    # ניקוי התוכן
    cleaned = clean_congressional_record(content)
    
    # שמירה לקובץ חדש
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    except Exception as e:
        print(f"שגיאה בכתיבה לקובץ '{output_file_path}': {e}")
        return
    
    # הדפסת סיכום עבור הקובץ
    print(f"✓ נוקה: {os.path.basename(input_file_path)}")
    print(f"   אורך מקורי: {len(content):,} תווים")
    print(f"   אורך אחרי ניקוי: {len(cleaned):,} תווים")
    print(f"   נחסכו: {len(content) - len(cleaned):,} תווים ({100 * (1 - len(cleaned)/len(content)):.1f}%)")
    print("-" * 40)

def process_directory(input_dir, output_dir, prefix):
    """
    מבצע ניקוי על כל הקבצים בתיקייה שמתחילים בקידומת נתונה.
    """
    print(f"מתחיל עיבוד בתיקייה: {input_dir}")
    
    # 1. יצירת תיקיית יעד אם אינה קיימת
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"תיקיית יעד: {output_dir}")
    
    # 2. חיפוש קבצים מתאימים
    search_path = os.path.join(input_dir, f'{prefix}*.txt')
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        print(f"❌ לא נמצאו קבצים בנתיב '{search_path}'. ודא שהתיקייה והקידומת נכונים.")
        return

    print(f"🎉 נמצאו {len(file_paths)} קבצים לעיבוד.")
    print("=" * 40)

    # 3. עיבוד כל קובץ
    for input_file_path in file_paths:
        file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        clean_file(input_file_path, output_file_path)

    print("=" * 40)
    print("✅ סיום העיבוד.")


if __name__ == "__main__":
    
    # הגדרות עיבוד
    INPUT_DIRECTORY = 'allData'
    OUTPUT_DIRECTORY = 'allData_cleaned'
    FILE_PREFIX = 'US_'
    
    # הפעלת עיבוד התיקייה
    process_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY, FILE_PREFIX)