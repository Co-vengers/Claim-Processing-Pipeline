from langchain_openai import ChatOpenAI
import os
import json


def run_bill_agent(pages: list[str]) -> dict:
    """
    Takes a list of page texts that were classified as itemized bills.
    Returns a dict with extracted billing information including line items.
    Returns an empty dict if no pages were given.
    """

    if not pages:
        print("[Bill Agent] No itemized bill pages found.")
        return {}

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

    combined_text = "\n\n---\n\n".join(pages)

    prompt = f"""You are an expert at extracting billing information from medical itemized bills.

Extract the following fields from the text below.
Return ONLY a valid JSON object with these exact keys:
- items: a list of objects, each with:
    - description (string): name of the service/item
    - quantity (number or null)
    - unit_price (number or null)
    - amount (number): total cost for this line item
- subtotal (number or null)
- taxes (number or null)
- total_amount (number): the final total bill amount
- currency (string): e.g. "INR" or "USD", if found

If a field is not found, use null.
For items, return an empty list [] if none found.
Do NOT include any explanation or markdown — only the JSON object.

Document text:
\"\"\"
{combined_text[:4000]}
\"\"\"
"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        error_text = str(e)
        if "403" in error_text or "Provider returned error" in error_text:
            print(
                f"[Bill Agent] Primary model '{primary_model}' failed with provider error. "
                f"Retrying with '{fallback_model}'."
            )
            response = build_llm(fallback_model).invoke(prompt)
        else:
            raise
    raw = response.content.strip()

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Bill Agent] Could not parse JSON response: {raw}")
        return {"raw_response": raw}
