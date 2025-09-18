# OCR Proof of Concept

This demo uses Tesseract and OpenCV to detect text in live video frames.  
It draws boxes around detected words and prints them with confidence scores.

---

## Requirements
- Python 3.x  
- OpenCV
- Pytesseract  
- Tesseract OCR installed and added to PATH  
  - (or just edit this line in the code to your path:  
    `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`)

---

## How to Run
1. Clone the repo and go to this folder
2. Install the dependencies above
3. Run the script by clicking "Run" in your editor or in terminal: python OCRDemo.py
