import requests

with open("final_image_protected.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/process",
        data={"claim_id": "CLAIM001"},
        files={"file": f}
    )
print(response.json())