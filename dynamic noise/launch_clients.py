import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# Path to the client.py script
client_script = "./client.py"

# Number of clients you want to launch
num_clients = 2

# Define max_workers based on available resources (e.g., limit to 2 clients at a time)
max_workers = 2

# Use ThreadPoolExecutor to run clients in parallel but control resource contention
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for i in range(num_clients):
        print(f"Launching client {i}")
#        futures.append(executor.submit(subprocess.run, ["python", client_script, str(i)]))
        # Replace subprocess.run with subprocess.Popen to avoid blocking
        futures.append(executor.submit(subprocess.Popen, ["python", client_script, str(i), str(num_clients)]))

        
        # Introduce a delay between launching clients
#        time.sleep(2)  # Wait for 2 seconds before launching the next client

    # Wait for all clients to complete
    for future in futures:
        future.result()


