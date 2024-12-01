import time
import psutil
import os
from PIL import Image
import torch
import numpy as np

def gpu_grayscale_conversion(input_image_path, output_image_path):
    # Start time and resource tracking
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    initial_cpu = psutil.cpu_percent(interval=None)

    # Load image and convert to a PyTorch tensor, then move it to GPU
    image = Image.open(input_image_path).convert('RGB')
    image_np = torch.from_numpy(np.array(image)).float().cuda()  # Move image to GPU and convert to float

    # Separate RGB channels and calculate grayscale using the luminosity formula
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    grayscale_gpu = (0.2989 * r + 0.5870 * g + 0.1140 * b).to(torch.uint8)  # Convert back to uint8

    # Move the grayscale image back to CPU
    grayscale_cpu = grayscale_gpu.cpu().numpy()

    # Convert to PIL image and save
    grayscale_image = Image.fromarray(grayscale_cpu, mode='L')
    grayscale_image.save(output_image_path)

    # End time and resource tracking
    end_time = time.time()
    final_memory = process.memory_info().rss
    final_cpu = psutil.cpu_percent(interval=None)

    # Calculations
    time_taken = end_time - start_time
    memory_used = (final_memory - initial_memory) / (1024 * 1024)  # Convert bytes to MB

    # Log details
    with open("parallel_gpu_log.txt", "w") as log_file:
        log_file.write(f"Time taken for GPU grayscale conversion: {time_taken:.4f} seconds\n")
        log_file.write(f"Initial CPU usage: {initial_cpu}%\n")
        log_file.write(f"Final CPU usage: {final_cpu}%\n")
        log_file.write(f"Initial memory usage: {initial_memory / (1024 * 1024):.4f} MB\n")
        log_file.write(f"Final memory usage: {final_memory / (1024 * 1024):.4f} MB\n")
        log_file.write(f"Memory used during execution: {memory_used:.4f} MB\n")

    print("GPU grayscale conversion complete. Log file and output image saved.")

# Ensure this is only run if the script is the main module
if __name__ == '__main__':
    input_image_path = 'pic.jpg'  # Replace with your actual image path
    output_image_path = 'output_parallel_gpu_grayscale.jpg'
    gpu_grayscale_conversion(input_image_path, output_image_path)
