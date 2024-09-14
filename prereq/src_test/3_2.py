import time
import math
from multiprocessing import Pool
from utils import est_util
import json
import time
import os

"""
Here I am having a main process where I am defining the arguments for the other processes.
Also from here, I am spawning 2 parallel processes, with some phi difference.
"""
process_start_time = 0
process_time = 0
cpu_util = []
times = []

def get_active_idle_time(start_time, phi, d, amplitude, period):
    phi_radian = phi * math.pi
    sine_wave = d + amplitude * math.sin((2 * math.pi * ((time.time() - start_time) % period / period)) + phi_radian)
    active_period = (sine_wave - (d - amplitude)) / (2 * amplitude)
    idle_period = 1 - active_period
    return active_period, idle_period


def simulate_sinusoidal_cpu_load(duration, start_time, phi, kwargs):
    global process_start_time
    global process_time
    global cpu_util
    pid = os.getpid()
    while (time.time() - start_time) < duration:
        active_period, idle_period = get_active_idle_time(start_time, phi,  **kwargs)
        end_time = time.time() + 0.1
        process_start_time = time.time()
        while time.time() < end_time:
            start_cpu_time = time.process_time()
            current_load_time = time.time()
            while(time.time() - current_load_time) < 0.1 * active_period:
                pass
            time.sleep(0.1 * idle_period)
            process_time = time.process_time() - start_cpu_time
            cpu_util.append(est_util(process_start_time, process_time))
            times.append(time.time() - start_time)
    data = {
        "times": times,
        "util": cpu_util
    }
    with open(f"cpu_util_{pid}.json", "w") as f:
        json.dump(data, f)
    
    

def get_multi_cpu_load(num_waves, phi_diff):
    duration = 50
    start_time = time.time()
    d = 1
    amplitude = 1
    period = 5
    kwargs = {
        "d": d, 
        "amplitude": amplitude, 
        "period": period,
    }
    phi_diff_list = [phi_diff * x for x in list(range(0, num_waves))]
    args = [(duration, start_time, phi_diff_list[i], kwargs) for i in range(0, num_waves)]
    with Pool(2) as p:
        p.starmap(simulate_sinusoidal_cpu_load, args) # using multiprocessing pool to run 2 processes so ideally they should be running in different cores

if __name__ == "__main__":
    get_multi_cpu_load(2, 1)


