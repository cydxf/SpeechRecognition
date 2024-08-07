import concurrent.futures
import speech_recognition as sr
from pydub import AudioSegment
from tqdm import tqdm
import subprocess
import os


def extract_audio(file_path):
    """
    从输入文件中提取音频，支持 wav、mp4 和 mp3 文件。
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    audio_path = "temp_audio.wav"

    if file_extension == ".wav":
        return file_path
    elif file_extension in [".mp4", ".mp3"]:
        # 构建 ffmpeg 命令
        command = ['ffmpeg', '-i', file_path, audio_path]
        print(f"运行命令: {' '.join(command)}")  # 打印命令以供调试
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
            print(result.stderr.decode())
        except FileNotFoundError:
            print("找不到 ffmpeg 可执行文件，请确保 ffmpeg 已正确安装并在系统路径中。")
            raise
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg 命令执行失败: {e}")
            raise

        if os.path.exists(audio_path):
            print(f"音频成功提取到: {audio_path}")
            return audio_path
        else:
            print("音频提取失败，未生成目标文件。")
            raise FileNotFoundError("音频提取失败，未生成目标文件。")
    else:
        raise ValueError("不支持的文件格式。请提供 wav、mp4 或 mp3 文件。")


def transcribe_audio_segment(segment_path):
    """
    将音频片段转换为文本。
    """
    recognizer = sr.Recognizer()

    with sr.AudioFile(segment_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            return text
        except sr.UnknownValueError:
            return "Google 语音识别无法理解音频"
        except sr.RequestError as e:
            return f"无法从 Google 语音识别服务请求结果; {e}"


def split_audio(audio_path, segment_length=30000, buffer_length=1000):
    """
    将音频文件分割成较短片段。
    """
    audio = AudioSegment.from_wav(audio_path)
    print(f"音频长度: {len(audio)} 毫秒")

    segments = []
    for i in range(0, len(audio), segment_length):
        start = max(0, i - buffer_length)
        end = min(len(audio), i + segment_length + buffer_length)
        segment = audio[start:end]
        segment_path = f"temp_segment_{i}.wav"
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
        print(f"生成片段: {segment_path}, 长度: {len(segment)} 毫秒")

    return segments


def transcribe_long_audio(file_path):
    """
    将长音频文件分割成较短片段并转换为文本。
    """
    audio_file = extract_audio(file_path)
    segment_paths = split_audio(audio_file)

    transcription = ""

    # 使用并行处理音频片段
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(transcribe_audio_segment, segment_path): segment_path for segment_path in
                   segment_paths}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="转换音频片段"):
            transcription += future.result() + " "

    # 删除临时片段文件
    for segment_path in segment_paths:
        if os.path.exists(segment_path):
            os.remove(segment_path)
            print(f"已删除临时片段: {segment_path}")
        else:
            print(f"找不到临时片段: {segment_path}")

    return transcription


def save_transcription(text, output_file):
    """
    将转换后的文本保存到文件。
    """
    with open(output_file, "w") as file:
        file.write(text)


def main(input_file, output_file):
    """
    主函数，处理输入文件并保存转换结果。
    """
    transcription = transcribe_long_audio(input_file)
    save_transcription(transcription, output_file)
    print(f"转换结果已保存到 {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 wav、mp4 和 mp3 文件中的音频转换为文本。")
    parser.add_argument("input_file", type=str, help="输入音频或视频文件路径。")
    parser.add_argument("output_file", type=str, help="输出文本文件路径。")

    args = parser.parse_args()

    main(args.input_file, args.output_file)
