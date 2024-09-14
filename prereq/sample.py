import time
import math

def simulate_cpu_load_sine_wave(duration, max_load=100, period=5):
    """
    Simulates CPU load in a sine wave pattern.

    :param duration: Total duration to maintain the load in seconds.
    :param max_load: Maximum load percentage during the peak of the sine wave.
    :param period: Period of the sine wave in seconds.
    """
    start_time = time.time()
    interval = 0.5  # Interval for each calculation cycle in seconds
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        # Calculate the current load percentage using sine wave
        load_percentage = (math.sin(2 * math.pi * elapsed / period) + 1) / 2 * max_load
        
        # Busy work period
        busy_start = time.time()
        while time.time() - busy_start < interval * (load_percentage / 100):
            math.sqrt(12345)  # CPU intensive operation

        # Sleep period
        time.sleep(max(0, interval - (time.time() - busy_start)))

if __name__ == "__main__":
    duration = 60  # Total duration in seconds
    max_load = 50  # Max load percentage
    period = 5     # Sine wave period in seconds
    simulate_cpu_load_sine_wave(duration, max_load, period)