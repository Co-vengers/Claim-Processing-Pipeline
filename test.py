import requests

with open("final_image_protected.pdf", "rb") as f:
    response = requests.post(
        "https://claim-processing-pipeline-rnz2.onrender.com/api/process",
        data={"claim_id": "CLAIM001"},
        files={"file": f}
    )
print(response.json())