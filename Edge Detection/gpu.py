import numpy as np
from numba import cuda
import math
import cv2
import time
import psutil
import os

@cuda.jit
def sobel_filter(input_image, output_image):
    x, y = cuda.grid(2)
    
    if x < input_image.shape[0] - 2 and y < input_image.shape[1] - 2:
        gx = (input_image[x, y] - input_image[x + 2, y] +
              2 * input_image[x, y + 1] - 2 * input_image[x + 2, y + 1] +
              input_image[x, y + 2] - input_image[x + 2, y + 2])
              
        gy = (input_image[x, y] - input_image[x, y + 2] +
              2 * input_image[x + 1, y] - 2 * input_image[x + 1, y + 2] +
              input_image[x + 2, y] - input_image[x + 2, y + 2])
        
        output_image[x + 1, y + 1] = math.sqrt(gx**2 + gy**2)

def apply_sobel_filter(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Check if the image was loaded
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Measure the start time
    start_time = time.time()

    # Track CPU and memory usage before processing
    process = psutil.Process(os.getpid())
    initial_cpu_usage = process.cpu_percent(interval=None)
    initial_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Create CUDA device array for the image
    d_image = cuda.to_device(image)

    # Allocate memory for the output image
    output_image = np.zeros_like(image)
    d_output_image = cuda.to_device(output_image)

    # Define block and grid dimensions
    threadsperblock = (16, 16)
    blockspergrid_x = (image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Apply Sobel filter using CUDA
    sobel_filter[blockspergrid, threadsperblock](d_image, d_output_image)

    # Copy the result back to the CPU
    output_image = d_output_image.copy_to_host()

    # Measure end time and calculate total time taken
    end_time = time.time()
    time_taken = end_time - start_time

    # Track CPU and memory usage after processing
    final_cpu_usage = process.cpu_percent(interval=None)
    final_memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Normalize and convert to uint8 for saving
    output_image = np.clip(output_image / output_image.max() * 255, 0, 255).astype(np.uint8)
    
    # Save the processed image
    output_path = "edge_detected_output.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"Edge-detected image saved as '{output_path}'")

    # Prepare log content
    log_content = (
        "GPU EXECUTION\n"
        f"Time taken for edge detection: {time_taken:.2f} seconds\n"
        f"Initial CPU usage: {initial_cpu_usage:.2f}%\n"
        f"Final CPU usage: {final_cpu_usage:.2f}%\n"
        f"Initial memory usage: {initial_memory_usage:.2f} MB\n"
        f"Final memory usage: {final_memory_usage:.2f} MB\n"
        f"Memory used during execution: {final_memory_usage - initial_memory_usage:.2f} MB\n"
        "--------------------------------------------------\n"
    )

    # Write the log to file
    with open("gpu_execution_log.txt", "a") as log_file:
        log_file.write(log_content)

    print("Execution statistics saved in 'gpu_execution_log.txt'")

# Example usage
apply_sobel_filter('pic.jpg')  # Replace with the path to your image
