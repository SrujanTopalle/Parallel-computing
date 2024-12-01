from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment
from pydub.silence import detect_silence
import os
import time
import psutil
import torch

def extract_audio(vid_name, audio_output_path):
    video = VideoFileClip(vid_name)
    audioclip = video.audio
    audioclip.write_audiofile(audio_output_path)

def detecting_silence(audio):
    silent_sections = detect_silence(
        audio, 
        min_silence_len=800,  
        silence_thresh=-75     
    )
    silent_sections = [(start / 1000, end / 1000) for start, end in silent_sections]
    return silent_sections

def cut_video(input_video_path, silence_intervals, output_video_path):
    video = VideoFileClip(input_video_path)
    non_silence_intervals = []
    previous_end = 0.0
    for start, end in silence_intervals:
        if previous_end < start:
            non_silence_intervals.append((previous_end, start))
        previous_end = end
    if previous_end < video.duration:
        non_silence_intervals.append((previous_end, video.duration))
    clips = [video.subclip(start, end) for start, end in non_silence_intervals]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    video.close()
    final_clip.close()

def delete_audio_file(audio_output_path):
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)

def log_performance(log_file, start_time, end_time):
    execution_time = end_time - start_time
    memory_info = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU"
    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2) if gpu_available else 0
    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2) if gpu_available else 0

    with open(log_file, 'w') as log:
        log.write(f"Execution Time: {execution_time:.2f} seconds\n")
        log.write(f"CPU Usage: {cpu_usage}%\n")
        log.write(f"Memory Usage: {memory_info.percent}%\n")
        log.write(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB\n")
        log.write(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB\n")
        log.write(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB\n")
        log.write(f"GPU Used: {gpu_name}\n")
        log.write(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB\n")
        log.write(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB\n")
        log.write("SUCCESS!\n")

if __name__ == '__main__':
    start = time.time()

    vid_name = 'name.mp4'
    audio_output_path = 'output_audio.mp3'
    video_output_path = 'output_video.mp4'
    log_file = 'performance_log.txt'

    extract_audio(vid_name, audio_output_path)
    audio = AudioSegment.from_file(audio_output_path)
    silence_list = detecting_silence(audio)
    cut_video(vid_name, silence_list, video_output_path)
    delete_audio_file(audio_output_path)

    end = time.time()
    log_performance(log_file, start, end)
