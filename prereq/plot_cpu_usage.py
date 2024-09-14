import json
import matplotlib.pyplot as plt

def plot_est_util():
    cpu_util = json.load(open("cpu_utils_3.json"))
    times = cpu_util["times"]
    utilizations = cpu_util["util"]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, utilizations, label='Estimated CPU Utilization (%)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Utilization (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print(f"Plotting estimated CPU utilization")
    plot_est_util()