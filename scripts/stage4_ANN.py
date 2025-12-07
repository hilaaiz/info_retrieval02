import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import matplotlib.pyplot as plt

# --- 0. הגדרות נתיבים ופרמטרים קריטיים ---
# הנחה: קבצים אלו נמצאים בתיקייה הראשית INFO_RETRIEVAL02
METADATA_PATH = Path('documents_metadata.csv') 
Y_LABELS_PATH = Path('y_labels_num.npy') 

MAX_VOCAB_SIZE = 20000  # גודל אוצר המילים
MAX_SEQUENCE_LENGTH = 500  
EMBEDDING_DIM = 100       

# --- 1. טעינת נתונים והכנה ל-Embedding ---
try:
    # טעינת טקסטים מקוריים (X) מ-documents_metadata.csv
    metadata = pd.read_csv(METADATA_PATH)
    X_text = metadata['text'].values
    
    # --- יצירת התוויות (Y) בצורה בטוחה מחדש מתוך metadata ---
    label_map = {"UK": 0, "US": 1}
    y_num = metadata["country"].map(label_map).to_numpy() 
    
    # הגדרת שמות ומספר המחלקות
    num_classes = len(np.unique(y_num)) # = 2
    final_class_names = ['UK', 'US'] # הגדרה קשיחה לדו"ח הסיווג
    
except FileNotFoundError as e:
    print(f"שגיאה קריטית: קובץ ה-metadata לא נמצא בנתיב: {e}. ודא שהרצת את Stage 1.")
    exit()

print(f'Number of documents: {len(X_text)}')
print(f'Number of classes: {num_classes}')


# 2. Tokenization: הפיכת טקסטים לרצפי אינדקסים (יצירת המילון המשותף)
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(X_text)
X_sequences = tokenizer.texts_to_sequences(X_text)

# 3. Padding: הבטחת אורך קבוע לכל רצף
X_padded = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# 4. המרת y ל-One-Hot Encoding
y_one_hot = tf.keras.utils.to_categorical(y_num, num_classes=num_classes)


# --- 5. חלוקת נתונים (Train/Val/Test) [cite: 65-67] ---

# 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_padded, y_one_hot, test_size=0.2, random_state=42, stratify=y_num
)

# מתוך ה-80%: 90% train, 10% val (0.10 / 0.80 = 0.125)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val.argmax(axis=1)
)

print(f"\nגודל קבוצות:")
print(f"  למידה (Train): {len(X_train)} ({len(X_train)/len(X_padded):.1%})")
print(f"  וולידציה (Validation): {len(X_val)} ({len(X_val)/len(X_padded):.1%})")
print(f"  בחינה (Test): {len(X_test)} ({len(X_test)/len(X_padded):.1%})")


# --- 6. הגדרת פונקציות (GELU ו-Model Builder) [cite: 73-84] ---

def gelu(x):
    # פונקציית האקטיבציה GELU
    if hasattr(tf.nn, 'gelu'):
        return tf.nn.gelu(x)
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def create_model(vocab_size, input_length, embedding_dim, num_classes, activation_func):
    """בניית מודל ANN בטופולוגיה הנדרשת עם שכבת Embedding."""
    model = Sequential()
    
    # שכבה ראשונה: Embedding layer 
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=input_length))
    
    # Flatten: הופך את פלט ה-Embedding לוקטור בודד
    model.add(Flatten())

    # שכבה שנייה: Hidden layer עם 10 קודקודים [cite: 75, 81]
    model.add(Dense(10, activation=activation_func))
    
    # שכבה שלישית: Hidden layer עם 10 קודקודים [cite: 76, 82]
    model.add(Dense(10, activation=activation_func))
    
    # שכבה רביעית: Hidden layer עם 7 קודקודים [cite: 77, 83]
    model.add(Dense(7, activation=activation_func))
    
    # שכבה אחרונה: Activation layer עם softmax [cite: 78, 84]
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- 7. אימון והערכה ---

# יצירת מודלים
input_length = X_padded.shape[1] 
model_relu = create_model(MAX_VOCAB_SIZE, input_length, EMBEDDING_DIM, num_classes, 'relu')
model_gelu = create_model(MAX_VOCAB_SIZE, input_length, EMBEDDING_DIM, num_classes, gelu)

# הגדרת Callbacks [cite: 70-72]
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
checkpoint_relu = ModelCheckpoint('best_ann_model_relu.keras', monitor='val_accuracy', save_best_only=True)
checkpoint_gelu = ModelCheckpoint('best_ann_model_gelu.keras', monitor='val_accuracy', save_best_only=True)


# אימון מודל ReLU [cite: 68-69]
print("\n" + "="*50)
print("אימון מודל ANN עם ReLU")
print("="*50)
history_relu = model_relu.fit(
    X_train, y_train, 
    epochs=15,               
    batch_size=16,           
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint_relu], 
    verbose=1
)

# אימון מודל GELU [cite: 68-69]
print("\n" + "="*50)
print("אימון מודל ANN עם GELU")
print("="*50)
history_gelu = model_gelu.fit(
    X_train, y_train, 
    epochs=15,               
    batch_size=16,           
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint_gelu], 
    verbose=1
)

# --- 8. הערכה ודו"ח סיווג ---
print("\n" + "="*50)
print("סיכום והערכה על קבוצת הבחינה")
print("="*50)

# טעינת המודלים הטובים ביותר
best_relu_model = tf.keras.models.load_model('best_ann_model_relu.keras', custom_objects={'gelu': gelu})
best_gelu_model = tf.keras.models.load_model('best_ann_model_gelu.keras', custom_objects={'gelu': gelu})

# הערכה על קבוצת הבחינה (Test Set)
_, acc_relu = best_relu_model.evaluate(X_test, y_test, verbose=0)
_, acc_gelu = best_gelu_model.evaluate(X_test, y_test, verbose=0)

if acc_relu > acc_gelu:
    best_model = best_relu_model
    model_name = "ReLU"
else:
    best_model = best_gelu_model
    model_name = "GELU"

print(f"ReLU Model Test Accuracy: {acc_relu:.4f}")
print(f"GELU Model Test Accuracy: {acc_gelu:.4f}")
print(f"\nהמודל הנבחר לדו\"ח הסיווג: {model_name}")

y_pred_one_hot = best_model.predict(X_test, verbose=0)
y_pred_class = np.argmax(y_pred_one_hot, axis=1) 
y_true_class = np.argmax(y_test, axis=1)

print("\n--- דו\"ח סיווג (Precision, Recall, F1, Accuracy) ---")
# שימוש ב-final_class_names המתוקן
print(classification_report(y_true_class, y_pred_class, target_names=final_class_names, digits=4))

# --- 9. ויזואליזציה (גרפי אימון) ---
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# יצירת גרפים
plot_training_history(history_relu, "ReLU Model")
plot_training_history(history_gelu, "GELU Model")