import time
import psutil
from PIL import Image
import os

# Load the image
input_image_path = 'pic.jpg'  # Replace with your actual image path
output_image_path = 'output_sequential_grayscale.jpg'

# Start time and resource tracking
start_time = time.time()
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss
initial_cpu = psutil.cpu_percent(interval=None)

# Sequential grayscale conversion
image = Image.open(input_image_path)
grayscale_image = image.convert('L')
grayscale_image.save(output_image_path)

# End time and resource tracking
end_time = time.time()
final_memory = process.memory_info().rss
final_cpu = psutil.cpu_percent(interval=None)

# Calculations
time_taken = end_time - start_time
memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert bytes to MB

# Log details
with open("sequential_log.txt", "w") as log_file:
    log_file.write(f"Time taken for grayscale conversion: {time_taken:.4f} seconds\n")
    log_file.write(f"Initial CPU usage: {initial_cpu}%\n")
    log_file.write(f"Final CPU usage: {final_cpu}%\n")
    log_file.write(f"Initial memory usage: {initial_memory / (1024 * 1024):.4f} MB\n")
    log_file.write(f"Final memory usage: {final_memory / (1024 * 1024):.4f} MB\n")
    log_file.write(f"Memory used during execution: {memory_used:.4f} MB\n")

print("Sequential grayscale conversion complete. Log file and output image saved.")
