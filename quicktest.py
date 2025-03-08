import os
import time
import yt_dlp
from pydub import AudioSegment
from ailabs_asr.streaming import StreamingClient

# 設定雅婷的 API 金鑰
API_KEY = ''  # 請替換成你的正確 API Key

# 下載 YouTube 影片的音訊
def download_audio(youtube_url, output_dir="downloads"):
    """ 下載 YouTube 影片音訊並轉換為 MP3 """
    
    # 確保資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 產生唯一的檔名（使用時間戳）
    timestamp = int(time.time())  
    output_path = os.path.join(output_dir, f"audio_{timestamp}.mp3")
    
    # `yt-dlp` 下載選項
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace(".mp3", ""),  # 確保不會出現雙重 .mp3
    }

    # 開始下載
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # 確認下載成功的檔案
    if os.path.exists(output_path):
        return output_path
    elif os.path.exists(output_path + ".mp3"):  # 檢查是否有雙重 .mp3
        return output_path + ".mp3"
    else:
        raise FileNotFoundError(f"下載失敗，找不到音訊檔案：{output_path}")

# 將音訊轉換為 WAV 格式
def convert_to_wav(input_audio, output_wav='audio.wav'):
    """ 將 MP3 轉換為 WAV 格式 """
    audio = AudioSegment.from_file(input_audio)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 設定為單聲道，16kHz
    audio.export(output_wav, format='wav')
    return output_wav

# 處理即時轉錄結果的回調函數
def on_processing_sentence(message):
    print(f"[即時轉錄結果]: {message['asr_sentence']}")

# 處理最終轉錄結果的回調函數
def on_final_sentence(message):
    print(f"[最終轉錄結果]: {message['asr_sentence']}")

# 語音識別主函數
def transcribe_audio(wav_file):
    """ 使用雅婷 ASR 進行語音識別 """
    asr_client = StreamingClient(API_KEY)
    asr_client.start_streaming_wav(
        pipeline='asr-zh-tw-std',  # 使用支援中文和台語的模型
        file=wav_file,
        verbose=False,
        on_processing_sentence=on_processing_sentence,
        on_final_sentence=on_final_sentence
    )

if __name__ == '__main__':
    youtube_url = input("請輸入 YouTube 影片網址：")

    print("正在下載 YouTube 音訊...")
    mp3_file = download_audio(youtube_url)  # ✅ 修正函數名稱

    print("正在轉換音訊為 WAV 格式...")
    wav_file = convert_to_wav(mp3_file)

    print("正在進行語音識別...")
    transcribe_audio(wav_file)

    print("✅ 完成！")
