from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from utils.pdf_utils import extract_pages
from graph import compiled_graph

app = FastAPI(
    title="Claim Processing Pipeline",
    description="Processes insurance claim PDFs using LangGraph + OpenRouter",
    version="1.0.0"
)


@app.get("/")
def root():
    """Health check — visit this in browser to confirm the server is running."""
    return {"status": "ok", "message": "Claim Processing API is running."}


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Main endpoint.
    Accepts:
      - claim_id: a string identifier for this claim
      - file: the PDF file to process

    Returns:
      JSON with all extracted data from the PDF
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    print(f"\n{'='*50}")
    print(f"Processing claim: {claim_id}  |  File: {file.filename}")
    print(f"{'='*50}")

    file_bytes = await file.read()

    pages = extract_pages(file_bytes)
    print(f"[main] Extracted {len(pages)} pages from PDF")

    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract any pages from the PDF.")

    initial_state = {
        "claim_id": claim_id,
        "pages": pages,
        "classified_pages": {},
        "identity_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_result": None
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        print(f"[main] Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    result = final_state.get("final_result", {})
    print(f"\n[main] Done! Returning result for claim: {claim_id}")

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
