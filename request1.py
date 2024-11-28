import requests
import base64

# API Endpoint
url = "http://127.0.0.1:8000/summarize/"
filename="C:/Users/SwastiShetty/Downloads/Pride and Prejudice Author Jane Austen.pdf"
# Convert a file to base64
with open(filename, "rb") as pdf_file:
    encoded_content = base64.b64encode(pdf_file.read()).decode("utf-8")
    print(encoded_content)
# Prepare the payload
payload = {
    "files": [
        {
            "file_name": filename,
            "content": str(encoded_content)
        }
    ]
}

# Send POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())
