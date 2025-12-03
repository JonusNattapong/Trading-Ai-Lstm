import requests
import json

# Check the dataset structure
url = 'https://huggingface.co/api/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/tree/main'
response = requests.get(url)
print('Status:', response.status_code)

if response.status_code == 200:
    data = response.json()
    print('Files:')
    for item in data:
        print(f" - {item['path']} ({item['type']})")

        # If it's a file, try to get its content
        if item['type'] == 'file' and item['path'].endswith('.csv'):
            file_url = f"https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/{item['path']}"
            file_response = requests.get(file_url)
            print(f"   File size: {len(file_response.content)} bytes")
            if len(file_response.content) < 1000:
                print(f"   Content preview: {file_response.text[:200]}...")
else:
    print('Error:', response.text)