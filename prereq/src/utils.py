import time

def est_util(process_start_time, process_time):
    elapsed_time = time.time() - process_start_time
    cpu_util = process_time/ elapsed_time
    return cpu_util

