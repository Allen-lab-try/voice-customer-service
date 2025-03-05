import os
import time
import yt_dlp
import whisper

def download_audio(youtube_url, output_dir="downloads"):
    """ 下載 YouTube 影片音訊並轉換為 MP3 """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = int(time.time())  # 取得時間戳
    output_path = os.path.join(output_dir, f"audio_{timestamp}.mp3")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': output_path,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)
        return output_path

def transcribe_audio(audio_file, model_size="large"):
    """ 使用 Whisper 進行語音轉錄 """
    print(f"正在使用 Whisper ({model_size}) 轉錄 {audio_file}...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file)
    return result["text"]

def main():
    youtube_url = input("請輸入 YouTube 影片網址: ")
    print("下載音訊中...")
    audio_file = download_audio(youtube_url)
    
    print("開始轉錄...")
    transcript = transcribe_audio(audio_file)
    
    output_file = audio_file.replace(".mp3", ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print("轉錄完成！結果已儲存於:", output_file)
    print("\n===== 轉錄內容 =====\n")
    print(transcript)

if __name__ == "__main__":
    main()
