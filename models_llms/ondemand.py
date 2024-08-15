# 
import requests

url = 'https://api.on-demand.io/chat/v1/sessions'
data = {
    "externalUserId": "06931013-9ae4-449d-87e8-4224508ba0d0",
    "pluginIds": []
}
headers = {
    'Content-Type': 'application/json',
    'apikey': 'OzETGcZoelR2nwtSFRnic3ITs89NoxjO'
}

response = requests.post(url, json=data, headers=headers)

print(response.json)

