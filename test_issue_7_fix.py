
import os
import sys

# Füge das src-Verzeichnis zum Pfad hinzu, damit wir importieren können
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from src.api.v1.rag.documents.upload import validate_file_type

def test_chinese_truncation():
    # Wir erzeugen eine Datei, bei der ein 3-Byte-Zeichen an der 512-Byte-Grenze abgeschnitten wird.
    # "你好" sind 6 Bytes.
    # 85 * "你好" = 510 Bytes.
    # Das nächste "你" (3 Bytes) würde bei 511, 512, 513 liegen.
    # Lesen wir 512 Bytes, haben wir 510 saubere Bytes + 2 Bytes Müll am Ende.
    
    content = "你好" * 100 
    filename = "test_chinese.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Testdatei erstellt: {filename}")
    
    try:
        # Test 1: Chinesische Datei (sollte True sein)
        result = validate_file_type(filename, ".md")
        if result:
            print("✅ SUCCESS: Chinesische Datei wurde korrekt erkannt (trotz 512-Byte-Grenze).")
        else:
            print("❌ FAILURE: Chinesische Datei wurde fälschlicherweise abgelehnt.")

        # Test 2: Echte Binärdatei (sollte False sein)
        # Wir schreiben ein Null-Byte rein, was in Textdateien nicht vorkommen sollte.
        with open("test_binary.md", "wb") as f:
            f.write(b"Hello World \x00 binary content")
        
        result_bin = validate_file_type("test_binary.md", ".md")
        if not result_bin:
            print("✅ SUCCESS: Binärdatei (mit Null-Byte) wurde korrekt abgelehnt.")
        else:
            print("❌ FAILURE: Binärdatei wurde fälschlicherweise akzeptiert.")

    finally:
        # Aufräumen
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists("test_binary.md"):
            os.remove("test_binary.md")

if __name__ == "__main__":
    test_chinese_truncation()
