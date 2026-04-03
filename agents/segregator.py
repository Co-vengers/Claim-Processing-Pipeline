from langchain_openai import ChatOpenAI
import os

VALID_DOC_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other"
]


def run_segregator(pages: list[dict]) -> dict:
    """
    Takes a list of pages (from pdf_utils.extract_pages).
    Classifies each page using an OpenRouter model.
    Returns a dict like:
    {
        "identity_document": ["page 1 text...", "page 3 text..."],
        "itemized_bill":      ["page 2 text..."],
        "discharge_summary":  ["page 4 text..."],
        ...
    }
    Only doc types that actually appear will be in the output.
    """

    def build_llm(model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0
        )

    primary_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/auto")
    llm = build_llm(primary_model)

    classified: dict[str, list[str]] = {}

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text:
            print(f"[Segregator] Page {page_number} is blank, skipping.")
            continue

        prompt = f"""You are a document classifier for insurance claim processing.

Classify the following page text into EXACTLY ONE of these document types:
{", ".join(VALID_DOC_TYPES)}

Rules:
- Reply with ONLY the document type label (e.g. "identity_document")
- No explanation, no punctuation, just the label
- If unsure, use "other"

Page text:
\"\"\"
{page_text[:3000]}
\"\"\"
"""

        try:
            response = llm.invoke(prompt)
        except Exception as e:
            error_text = str(e)
            # Retry once on provider-side denials by routing through a fallback model.
            if "403" in error_text or "Provider returned error" in error_text:
                print(
                    f"[Segregator] Primary model '{primary_model}' failed with provider error. "
                    f"Retrying with '{fallback_model}'."
                )
                response = build_llm(fallback_model).invoke(prompt)
            else:
                raise
        doc_type = response.content.strip().lower()

        if doc_type not in VALID_DOC_TYPES:
            print(f"[Segregator] Page {page_number}: unknown type '{doc_type}', using 'other'")
            doc_type = "other"

        print(f"[Segregator] Page {page_number} → {doc_type}")

        if doc_type not in classified:
            classified[doc_type] = []
        classified[doc_type].append(page_text)

    return classified
