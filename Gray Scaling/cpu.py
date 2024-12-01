import time
import psutil
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Define a function to convert a part of the image to grayscale
def process_chunk(chunk):
    return chunk.convert('L')

# Main function to handle parallel grayscale conversion
def parallel_grayscale_conversion(input_image_path, output_image_path):
    # Start time and resource tracking
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    initial_cpu = psutil.cpu_percent(interval=None)

    # Load and split the image into chunks for parallel processing
    image = Image.open(input_image_path)
    image_np = np.array(image)
    height, width, _ = image_np.shape
    num_chunks = os.cpu_count()  # Use the number of available CPU cores

    # Calculate chunk height
    chunk_height = height // num_chunks

    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        chunks = [image.crop((0, i * chunk_height, width, (i + 1) * chunk_height)) for i in range(num_chunks)]
        grayscale_chunks = list(executor.map(process_chunk, chunks))

    # Combine chunks
    grayscale_image = Image.new('L', (width, height))
    for i, chunk in enumerate(grayscale_chunks):
        grayscale_image.paste(chunk, (0, i * chunk_height))

    # Save the final grayscale image
    grayscale_image.save(output_image_path)

    # End time and resource tracking
    end_time = time.time()
    final_memory = process.memory_info().rss
    final_cpu = psutil.cpu_percent(interval=None)

    # Calculations
    time_taken = end_time - start_time
    memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert bytes to MB

    # Log details
    with open("parallel_cpu_log.txt", "w") as log_file:
        log_file.write(f"Time taken for parallel grayscale conversion: {time_taken:.4f} seconds\n")
        log_file.write(f"Initial CPU usage: {initial_cpu}%\n")
        log_file.write(f"Final CPU usage: {final_cpu}%\n")
        log_file.write(f"Initial memory usage: {initial_memory / (1024 * 1024):.4f} MB\n")
        log_file.write(f"Final memory usage: {final_memory / (1024 * 1024):.4f} MB\n")
        log_file.write(f"Memory used during execution: {memory_used:.4f} MB\n")

    print("Parallel CPU grayscale conversion complete. Log file and output image saved.")

# Ensure this is only run if the script is the main module
if __name__ == '__main__':
    input_image_path = 'pic.jpg'  # Replace with your actual image path
    output_image_path = 'output_parallel_cpu_grayscale.jpg'
    parallel_grayscale_conversion(input_image_path, output_image_path)
