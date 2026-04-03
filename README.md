# Claim Processing Pipeline

A FastAPI + LangGraph service that processes insurance claim PDFs using OpenRouter-hosted LLMs.

## How It Works

```
Upload PDF
    ↓
Segregator Agent  (OpenRouter model classifies each page into a document type)
    ↓         ↓              ↓
ID Agent  Discharge Agent  Bill Agent   (each gets only their relevant pages)
    ↓         ↓              ↓
         Aggregator
              ↓
         Final JSON
```

## Project Structure

```
claim_pipeline/
├── main.py                  # FastAPI app + /api/process endpoint
├── graph.py                 # LangGraph workflow (the flowchart in code)
├── agents/
│   ├── segregator.py        # Classifies pages into doc types using OpenRouter
│   ├── id_agent.py          # Extracts identity info
│   ├── discharge_agent.py   # Extracts diagnosis/hospital info
│   └── bill_agent.py        # Extracts billing line items
├── utils/
│   └── pdf_utils.py         # Reads PDF and extracts text per page
├── requirements.txt
└── .env                     # Your API key goes here (never commit this!)
```

## Setup Instructions

### 1. Get an OpenRouter API Key
- Go to https://openrouter.ai/
- Create an account and generate an API key
- Copy the key

### 2. Add your API settings to .env
```
OPENROUTER_API_KEY=your_actual_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python main.py
```
Server starts at: http://localhost:8000

### 5. Test the API

**Option A — Using the browser (Swagger UI):**
Open http://localhost:8000/docs
- Click `POST /api/process` → "Try it out"
- Fill in `claim_id` and upload your PDF
- Click Execute

**Option B — Using curl:**
```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLAIM001" \
  -F "file=@your_file.pdf"
```

**Option C — Using Python requests:**
```python
import requests

with open("your_file.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/process",
        data={"claim_id": "CLAIM001"},
        files={"file": f}
    )
print(response.json())
```

## Sample Output

```json
{
  "claim_id": "CLAIM001",
  "identity": {
    "patient_name": "John Doe",
    "date_of_birth": "1985-03-15",
    "id_number": "ABCD1234",
    "policy_number": "POL-9876",
    "insurance_provider": "Star Health",
    "gender": "Male"
  },
  "discharge_summary": {
    "diagnosis": "Acute Appendicitis",
    "admission_date": "2024-01-10",
    "discharge_date": "2024-01-13",
    "length_of_stay_days": 3,
    "treating_physician": "Dr. Sharma",
    "hospital_name": "City General Hospital",
    "treatment_summary": "Laparoscopic appendectomy performed successfully."
  },
  "itemized_bill": {
    "items": [
      { "description": "Room charges", "quantity": 3, "unit_price": 2000, "amount": 6000 },
      { "description": "Surgery charges", "quantity": 1, "unit_price": 25000, "amount": 25000 }
    ],
    "subtotal": 31000,
    "taxes": 1550,
    "total_amount": 32550,
    "currency": "INR"
  },
  "document_types_found": ["identity_document", "discharge_summary", "itemized_bill", "claim_forms"]
}
```

## Notes
- You can switch models by changing `OPENROUTER_MODEL` in `.env`
- Pages with no extractable text are skipped automatically
- If a document type is not found in the PDF, its section will be an empty `{}`
- Check the terminal logs to see each node's progress in real time
