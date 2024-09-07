# uvicorn 3_3_1:app --host 127.0.0.1 --port 8000
curl -X POST "http://127.0.0.1:8000/startProcess/" -H "Content-Type: application/json"
curl -X POST "http://127.0.0.1:5000/getUtil/" -H "Content-Type: application/json"
