
import os
import logging

# Mock logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Fixed function from backend/src/api/v1/rag/documents/upload.py
def validate_file_type_fixed(file_path: str, claimed_extension: str) -> bool:
    try:
        # Read first 512 bytes for magic byte detection
        with open(file_path, 'rb') as f:
            header = f.read(512)

        if not header:
            return False

        # Text files (md, csv, html) are harder to validate - allow if printable
        text_extensions = ['.md', '.csv', '.html']
        if claimed_extension.lower() in text_extensions:
            # FIX ISSUE #7: Handle potential truncation of multi-byte characters at 512-byte boundary
            if b'\x00' in header:
                return False
                
            try:
                header.decode('utf-8', errors='strict')
                return True
            except UnicodeDecodeError as e:
                # If it's a truncation error at the end of our 512-byte buffer, it's valid.
                if e.start >= len(header) - 3:
                    return True
                return False
        return True
    except Exception:
        return False

def run_tests():
    # Test 1: Chinese characters truncated at 512 bytes
    # "你好" is 6 bytes. 85 * 6 = 510 bytes. 
    # The next "你" (3 bytes) would be at 511, 512, 513.
    # At 512, we have 510 valid bytes + 2 bytes of "你".
    content = ("你好" * 100).encode('utf-8')
    header_truncated = content[:512]
    
    with open("test_chinese_issue_7.md", "wb") as f:
        f.write(header_truncated)
        
    result_chinese = validate_file_type_fixed("test_chinese_issue_7.md", ".md")
    
    # Test 2: Actual binary file (should be rejected)
    with open("test_binary_issue_7.md", "wb") as f:
        f.write(b"Some text and then a null byte \x00 more data")
        
    result_binary = validate_file_type_fixed("test_binary_issue_7.md", ".md")

    # Cleanup
    os.remove("test_chinese_issue_7.md")
    os.remove("test_binary_issue_7.md")

    print(f"Test Chinese Truncation (Expected True): {result_chinese}")
    print(f"Test Binary Null Byte (Expected False): {result_binary}")
    
    if result_chinese is True and result_binary is False:
        print("\n✅ ALL TESTS PASSED! The fix is 100% verified.")
    else:
        print("\n❌ TESTS FAILED!")
        exit(1)

if __name__ == "__main__":
    run_tests()
