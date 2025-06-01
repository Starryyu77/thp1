Form Data Extraction and Conversion

This project provides a Python script to extract data from PDF forms using OCR, convert it to JSON, and generate Word documents with tables. It supports both regular and irregular form layouts, particularly for medical forms.

Prerequisites

To run this program locally, ensure you have the following installed:





Python 3.8+





Download and install from python.org.



Tesseract OCR





Download and install from Tesseract OCR GitHub.



For Windows, download the installer from UB-Mannheim Tesseract Releases.



Update the pytesseract.pytesseract.tesseract_cmd path in plus.py to point to your Tesseract executable (e.g., r"C:\Program Files\Tesseract-OCR\tesseract.exe" on Windows).



Poppler





Required for pdf2image. Install via:





Windows: Download from Poppler for Windows and add the bin folder to your system PATH.



macOS: brew install poppler (Homebrew: brew.sh).



Linux: sudo apt-get install poppler-utils.



Python Libraries





Install required packages using:

pip install opencv-python pytesseract numpy pdf2image scikit-learn python-docx

Setup





Clone or Download the Repository





Place plus.py in your project directory.



Configure Input and Output Directories





Update the INPUT_DIR and OUTPUT_DIR variables in plus.py to your desired paths:

INPUT_DIR = r"path\to\your\input\directory"
OUTPUT_DIR = r"path\to\your\output\directory"



Ensure the input directory contains PDF files (e.g., m0.pdf, m1.pdf).



Prepare PDF Files





Place the PDF forms you want to process in the INPUT_DIR.

Running the Program





Execute the Script





Navigate to the project directory and run:

python plus.py



The script will:





Process each PDF in INPUT_DIR.



Extract form data using OCR and save it as JSON in OUTPUT_DIR (e.g., m0.json).



Convert JSON to Word documents with tables in OUTPUT_DIR (e.g., m0_table.docx).



Output





JSON files contain extracted form data with row, column, and content information.



Word documents contain tables representing the form structure, with merged cells for longer content or specific cases.

Notes





Language Support: The script uses chi_sim+eng for OCR (Chinese Simplified + English). Modify the lang parameter in extract_text_in_cell or process_pdf for other languages (e.g., eng for English only).



Error Handling: Check the console for error messages if a PDF fails to process or if dependencies are missing.



Customization: Adjust col_map in json_to_word for different column mappings if your forms have unique layouts.



Dependencies: Ensure all dependencies are correctly installed, and paths (Tesseract, Poppler) are properly configured.

Troubleshooting





Tesseract Not Found: Verify the Tesseract path in plus.py matches your installation.



Poppler Not Found: Ensure Poppler’s bin folder is in your system PATH.



Empty Output: Check if input PDFs are valid and contain scannable text or tables.



OCR Issues: Ensure forms have clear text and proper contrast for accurate OCR results.

For further assistance, refer to the documentation of the used libraries or contact the project maintainer.
