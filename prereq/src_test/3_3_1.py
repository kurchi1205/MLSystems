import math
import time
import os
from fastapi import FastAPI
from utils import est_util
import uvicorn
import logging
from threading import Thread

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

process_meta = {
    "process_start_time": 0,
    "process_time": 0
}

def get_active_idle_time(start_time, phi, d, amplitude, period):
    phi_radian = phi * math.pi
    sine_wave = d + amplitude * math.sin(2 * math.pi * ((time.time() - start_time) / period) + phi_radian)
    active_period = (sine_wave - (d - amplitude)) / (2 * amplitude)
    idle_period = 1 - active_period
    return active_period, idle_period

def simulate_sinusoidal_cpu_load(duration, start_time, phi, kwargs):
    global process_meta
    while True:
        active_period, idle_period = get_active_idle_time(start_time, phi, **kwargs)
        end_time = time.time() + 0.1
        process_start_time = time.time()
        while time.time() < end_time:
            current_load_time = time.time()
            start_cpu_time = time.process_time()
            while(time.time() - current_load_time) < active_period * 0.1:
                pass
            time.sleep(idle_period * 0.1)
            process_time = time.process_time() - start_cpu_time
            # updating meta for utilization calculation
            process_meta["process_time"] = process_time
            process_meta["process_start_time"] = process_start_time

app = FastAPI()

@app.post("/getUtil")
def get_utilization():
    logger.info(f"{process_meta}")
    cpu_util = est_util(process_meta["process_start_time"], process_meta["process_time"])
    return cpu_util

@app.on_event("startup")
def start_process_1():
    global process_meta
    duration = 30
    phi = 0
    d = 1
    amplitude = 1
    period = 5
    kwargs = {
        "d": d, 
        "amplitude": amplitude, 
        "period": period,
    }
    logger.info("Starting sine wave process")
    thread = Thread(target=simulate_sinusoidal_cpu_load, args=(duration, time.time(), phi, kwargs)) # to enable app startup 
    thread.start()

def start_server():
    uvicorn.run("3_3_1:app", port=5000, log_level="debug")

if __name__ == "__main__":
    start_server()