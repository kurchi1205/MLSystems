import math
import time
import requests

def get_t(time_period):
    response = requests.post("http://127.0.0.1:5000/getUtil/")   
    util = response.json()
    theta = math.asin(2 * util - 1) # getting the angle
    t = (theta/ (2 * math.pi)) * time_period # getting t from angle
    return t


def get_active_idle_time(phi, d, amplitude, period):
    t = get_t(period)
    phi_radian = phi * math.pi
    sine_wave = d + amplitude * math.sin((2 * math.pi * (t % period / period)) + phi_radian)
    active_period = (sine_wave - (d - amplitude)) / (2 * amplitude)
    idle_period = 1 - active_period
    return active_period, idle_period


def get_sine_wave(t, phase_diff):
    while True:
        d = 1
        amplitude = 1
        period = 5
        active_period, idle_period = get_active_idle_time(phase_diff, d, amplitude, period)
        end_time = time.time() + 0.1
        while time.time() < end_time:
            current_load_time = time.time()
            while(time.time() - current_load_time) < 0.1 * active_period:
                pass
            time.sleep(0.1 * idle_period)


if __name__=="__main__":
    try:
        time_period = 5
        phase_diff = 1
        t = get_t(time_period)
        get_sine_wave(t, phase_diff)
    except Exception as e:
        print("Server not running: ", e)


            

