from pdf2image import convert_from_bytes
import pytesseract


def extract_pages(file_bytes: bytes) -> list[dict]:
    """
    Takes raw PDF bytes.
    Converts each page to an image, then runs OCR to extract text.
    Returns a list of dicts: [{page_number, text}, ...]
    Works for both scanned PDFs and text-based PDFs.
    """

    images = convert_from_bytes(file_bytes, dpi=200)

    pages = []
    for i, image in enumerate(images):
        print(f"[pdf_utils] OCR processing page {i + 1}/{len(images)}...")

        text = pytesseract.image_to_string(image)
        text = text.strip()

        pages.append({
            "page_number": i + 1,
            "text": text
        })

    return pages