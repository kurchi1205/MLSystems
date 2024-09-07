import requests
import time
import json

start_time = time.time()
times = []
cpu_utils = []
while time.time() - start_time < 100:
    try:
        response = requests.post("http://127.0.0.1:5000/getUtil/")
        times.append(time.time() - start_time)
        cpu_utils.append(response.json())
    except:
        break

data = {
    "util": cpu_utils,
    "times": times
}
with open("cpu_utils_3.json", "w") as f:
    json.dump(data, f)