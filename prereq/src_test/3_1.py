import time
import math
from utils import est_util
import json

"""
Here I am having a main process where I am defining the arguments for the other processes.
"""
process_start_time = 0
process_time = 0
cpu_util = []
times = []

def get_active_idle_time(start_time, d, amplitude, period):
    sine_wave = d + amplitude * math.sin((2 * math.pi * ((time.time() - start_time) % period / period))) # adding a vertical shift because we need positive values for load
    active_period = (sine_wave - (d - amplitude)) / (2 * amplitude) #min max normalization
    idle_period = 1 - active_period # taking interval as 1
    return active_period, idle_period


def simulate_sinusoidal_cpu_load(duration, start_time, kwargs):
    global process_start_time
    global process_time
    global cpu_util
    while (time.time() - start_time) < duration: # running for some duration
        active_period, idle_period = get_active_idle_time(start_time, **kwargs)  # getting active and idle time
        end_time = time.time() + 0.1 # changing interval to 0.1
        process_start_time = time.time()
        while time.time() < end_time:
            start_cpu_time = time.process_time()
            current_load_time = time.time()
            while(time.time() - current_load_time) < 0.1 * active_period:
                pass # just creating some load
            time.sleep(0.1 * idle_period)
            process_time = time.process_time() - start_cpu_time
            cpu_util.append(est_util(process_start_time, process_time))
            times.append(time.time() - start_time)
    data = {
        "times": times,
        "util": cpu_util
    }
    # Saving cpu utilization for plotting.
    with open("cpu_util.json", "w") as f:
        json.dump(data, f)
    
    

def get_cpu_load():
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
    simulate_sinusoidal_cpu_load(duration, start_time, kwargs)

if __name__ == "__main__":
    get_cpu_load()


