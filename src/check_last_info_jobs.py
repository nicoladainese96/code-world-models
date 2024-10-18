import os
import re
import argparse
from datetime import datetime, timedelta

# Function to extract the last [INFO] timestamp from a file
def get_last_info_timestamp(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    last_info_time = None
    time_pattern = re.compile(r'\[INFO\] \[(\d{2}:\d{2}:\d{2})\]')
    
    for line in lines:
        match = time_pattern.search(line)
        if match:
            last_info_time = match.group(1)
    
    return last_info_time

# Function to compare timestamps
def check_if_time_exceeded(last_info_time, threshold_minutes, verbose=False):
    current_time = datetime.now().time()
    last_info_time = datetime.strptime(last_info_time, '%H:%M:%S').time()
    if verbose:
        print(last_info_time)
    time_diff = datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), last_info_time)
    return time_diff > timedelta(minutes=threshold_minutes)

# Main function to process job array
def process_job_array(job_id, array_size, threshold_minutes, base_path='./slurm', verbose=False):
    jobs_exceeding_threshold = []

    for a in range(array_size):
        filepath = os.path.join(base_path, f"{job_id}_{a}.out")
        
        if os.path.exists(filepath):
            last_info_time = get_last_info_timestamp(filepath)
            if last_info_time and check_if_time_exceeded(last_info_time, threshold_minutes):
                jobs_exceeding_threshold.append(a)
    
    return jobs_exceeding_threshold

# Entry point of the script
def main():
    parser = argparse.ArgumentParser(description="Check job logs for last [INFO] timestamp")
    parser.add_argument("job_id", type=str, help="Job ID")
    parser.add_argument("array_size", type=int, help="Size of the job array")
    parser.add_argument("--threshold_minutes", type=int, default=10, help="Threshold in minutes")
    parser.add_argument("--base_path", type=str, default="./slurm", help="Base path for job log files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    
    jobs_exceeding_threshold = process_job_array(args.job_id, args.array_size, args.threshold_minutes, args.base_path, args.verbose)
    if jobs_exceeding_threshold:
        print(f"Jobs exceeding the threshold: {jobs_exceeding_threshold}")
    else:
        print("No jobs exceeding the threshold.")

if __name__ == "__main__":
    main()
