# Mount the server with `uvicorn api:app`

import requests

query_input = {
    'user_id': 'new_user',
    'channel': 'Organic',
    'device_type': 'Desktop',
    'query_text': 'pie',
}
response = requests.post("http://127.0.0.1:8000/search/", json=query_input)
print(response.status_code)
print(response.json()['recommendations'][:10])