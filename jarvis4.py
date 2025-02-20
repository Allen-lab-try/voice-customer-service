import sounddevice as sd
import numpy as np
import edge_tts
import pvporcupine
import pyaudio
import asyncio
import struct
import os
import time
import subprocess  # ⬅ 用於關閉播放器
import whisperx  # ✅ WhisperX 替代 Whisper
from llama_cpp import Llama  # ✅ 使用本地 LLaMA 替代 OpenAI API
import torch
from collections import deque  # 🚀 用來存歷史對話記錄


# **🔹 加載 LLaMA 2 或 Mistral 7B 模型**
MODEL_PATH = r"D:\model\mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 請確認模型路徑正確

# ✅ 啟用 GPU 加速，減少 LLaMA 生成時間
llm = Llama(
    model_path=MODEL_PATH, 
    n_ctx=2048,  
    n_batch=512,  # 🚀 提高批量處理大小，加速推理
    n_gpu_layers=40  # 🚀 讓前 20 層運行在 GPU 上（如果有 GPU）
)


# **🔹 加載 WhisperX 模型**
device = "cuda" if torch.cuda.is_available() else "cpu"  # ✅ 自動選擇 GPU 或 CPU
whisperx_model = whisperx.load_model("medium", device)

buffer = []
MAX_BUFFER_SIZE = 16000 * 6  # 限制最大緩衝區為 5 秒
ACCESS_KEY = "w7yAkwffkJG/Odyqljlz/KZItUsA5YRioFz2vVB7F7rAxCfu4XkYxw=="  # 這裡填入你的 Porcupine API Key

is_speaking = False  # **用來標記 TTS 播放時是否應該靜音錄音**
player_process = None  # **用來存儲播放程式的進程**
awaiting_wake_word = True  # **控制是否要進行轉錄與對答**

# ✅ **加入對話歷史**
conversation_history = deque(maxlen=5)  # 🚀 只保存最近 5 句對話

# **🔹 1. LLaMA 解析語意並生成回應**
def llama_response(user_text):
    """ 使用本地 LLaMA 來生成回應 """
    response = llm(
        f"你是一個專業的語音客服助理，請提供精準的回應。\n使用者：{user_text}\n助手：",
        max_tokens=256,
        stop=["\n", "使用者："],  # 避免無限生成
        echo=False
    )
    return response["choices"][0]["text"].strip()

# **🔹 2. 語音合成（Edge TTS）**
async def text_to_speech(text):
    """ 使用 Edge TTS 轉語音並播放，確保錄音暫停，播放結束後自動關閉 """
    global is_speaking, player_process
    is_speaking = True  # **標記為 TTS 播放中，防止錄音**
    
    try:
        filename = "response.mp3"
        
        # **檢查檔案是否存在，避免覆蓋問題**
        if os.path.exists(filename):
            os.remove(filename)

        tts = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        await tts.save(filename)

        # **播放音訊**
        if os.name == "nt":  # Windows
            player_process = subprocess.Popen(["start", filename], shell=True)  # 開啟播放器
        elif os.name == "posix":  # macOS/Linux
            if "darwin" in os.sys.platform:
                player_process = subprocess.Popen(["afplay", filename])  # macOS
            else:
                player_process = subprocess.Popen(["mpg123", filename])  # Linux

        # **等待播放器結束**
        player_process.wait()

        # **關閉播放器**
        stop_audio_playback()

    except Exception as e:
        print(f"語音轉換錯誤: {e}")

    is_speaking = False  # **播放結束，恢復錄音**
    global awaiting_wake_word
    awaiting_wake_word = True  # **回到等待喚醒詞狀態**

def stop_audio_playback():
    """ 強制關閉播放程式，確保播放完畢後不影響錄音 """
    global player_process
    if player_process is not None:
        try:
            if os.name == "nt":  # Windows
                subprocess.run("taskkill /F /IM wmplayer.exe", shell=True)  # 關閉 Windows Media Player
            elif os.name == "posix":
                subprocess.run("pkill afplay", shell=True)  # macOS
                subprocess.run("pkill mpg123", shell=True)  # Linux
        except Exception as e:
            print(f"關閉播放器時發生錯誤: {e}")

# **🔹 3. 語音處理 `process_audio()`**
def process_audio(indata, frames, callback_time, status):
    global buffer, is_speaking, awaiting_wake_word

    if status:
        print(f"狀態錯誤：{status}")

    # **TTS 播放時不錄音**
    if is_speaking:
        return

    # 轉換數據格式
    audio_data = np.array(indata[:, 0], dtype=np.float32)
    buffer.extend(audio_data)

    print(f"🎤 目前錄音中，Buffer 長度: {len(buffer)}")

    # 限制 `buffer` 大小
    if len(buffer) > MAX_BUFFER_SIZE:
        print("🔹 偵測到完整語句，開始轉錄...")

        # **確保 buffer 轉換為正確格式**
        audio_input = np.array(buffer, dtype=np.float32)

        # **使用 WhisperX 進行轉錄**
        result = whisperx_model.transcribe(audio_input, batch_size=32, language="zh")

        recognized_text = " ".join([segment["text"] for segment in result["segments"]])

        print(f"🗣 WhisperX 轉錄：{recognized_text}")

        # **清空 buffer，防止重複轉錄**
        buffer.clear()

        # **檢查喚醒詞 "jarvis" 是否被提及**
        if "jarvis" not in recognized_text.lower():
            print("❌ 未偵測到喚醒詞 'jarvis'，不觸發回應。")
            return
        
        print(f"🗣️ 轉錄結果：{recognized_text}")
        response_text = llama_response(recognized_text)  
        print(f"🤖 LLaMA 回應：{response_text}")
        awaiting_wake_word = False  # **等待對話完成後再偵測新的喚醒詞**
        asyncio.run(text_to_speech(response_text))

# **🔹 4. 並行處理 LLaMA 和 TTS**
async def process_and_respond(user_text):
    """ ✅ LLaMA 生成 & Edge TTS 語音合成同時進行 """
    response_task = asyncio.create_task(llama_response(user_text))  # **LLaMA 生成**
    response_text = await response_task  

    print(f"🤖 LLaMA 回應：{response_text}")

    # ✅ **語音合成同時執行**
    speech_task = asyncio.create_task(text_to_speech(response_text))
    
    await speech_task

# **🔹 5. 開始監聽**
def start_conversation():
    print("🎤 等待喚醒詞 'jarvis'...")
    try:
        with sd.InputStream(callback=process_audio, channels=1, samplerate=16000, blocksize=2048):
            while True:
                sd.sleep(50)  # **減少 CPU 過度使用**
    except KeyboardInterrupt:
        print("\n🛑 偵測到 Ctrl+C，安全退出程式。")

# **🔹 5. 啟動程式**
if __name__ == "__main__":
    start_conversation()