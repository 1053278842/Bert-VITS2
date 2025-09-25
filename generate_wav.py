from webui import tts_fn
import utils
import os
import io
import numpy as np
from config import config
from infer import latest_version, get_net_g
from scipy.io.wavfile import write
import webui   # 注意：额外导入 webui 模块本身，用于注入全局变量
import requests
import soundfile as sf
from dotenv import load_dotenv
from rabbitmq_utils import rabbitmq_client
import json
import hashlib
from datetime import datetime, timedelta
import threading
import time

# 加载环境变量
load_dotenv()
url = os.environ.get("UPLOAD_API")
if url is None:
    print("未找到UPLOAD_API配置")
    raise Exception("UPLOAD_API 环境变量未配置！请参考ai-butler-api项目接口")


def init_tts():
    device = config.webui_config.device
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 初始化 hps 和 net_g
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )

    # 注入到 webui 的全局变量
    webui.hps = hps
    webui.net_g = net_g
    
def save_and_upload_tts(result,filename, folder="output"):
    status, audio_data = result
    if status != "Success":
        raise ValueError(f"TTS 失败: {status}")

    sr, audio_array = audio_data
    os.makedirs(folder, exist_ok=True)
    filename = f"{filename}.wav"
    file_path = os.path.join(folder, filename)

    # 保存到本地
    write(file_path, sr, audio_array)
    print(f"已保存音频到本地: {file_path}")

    # 上传到云端
    with open(file_path, "rb") as f:
        files = {"file": (filename, f, "audio/wav")}
        response = requests.post(url, files=files)

    print("上传状态码:", response.status_code)
    print("上传返回:", response.text)

    # 删除本地文件
    try:
        os.remove(file_path)
        print(f"已删除本地文件: {file_path}")
    except Exception as e:
        print(f"删除本地文件失败: {e}")
        

def tts(content):
    text = content
    speaker = "mxj"
    sdp_ratio = 0.3
    noise_scale = 0.6
    noise_scale_w = 0.9
    length_scale = 1
    language = "ZH"
    reference_audio = None
    emotion = "Happy"
    prompt_mode = "Text prompt"
    style_text = ""
    style_weight = 0.7

    return tts_fn(
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        prompt_mode,
        style_text,
        style_weight,
    )

def mq_callback(ch, method, properties, body):
    try:
        # 解析消息 JSON
        msg = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        print("消息不是有效的 JSON，跳过:", body)
        return

    # 检查 type 字段
    msg_type = msg.get("type")
    if msg_type != "normal" :
        print("消息 type 不符合，跳过:", msg_type)
        return

    # 获取 question 字段并生成哈希
    question = msg.get("question")
    if not question:
        print("消息没有 question 字段，跳过")
        return

    question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()
    
    
    # 获取 回答内容
    answer = msg.get("answer")

    
    # 获取 time 字段并解析
    time_str = msg.get("time")
    if not time_str:
        print("消息没有 time 字段，跳过")
        return

    try:
        sent_time = datetime.fromisoformat(time_str)
    except ValueError:
        print("time 字段格式不正确，跳过:", time_str)
        return

    # 判断是否在 1 小时内
    now = datetime.now()
    if now - sent_time > timedelta(hours=1):
        print(f"消息已过期 (>1h): {question}")
        return

    # 如果都符合条件，处理消息
    print(f"消息有效 ✅")
    print(f"问题哈希: {question_hash}")
    print(f"问题原文: {question}")
    print(f"发送时间: {time_str}")
    print(f"回答内容: {answer}")
    
    if not answer:
        answer ="好像...接收到转化内容哦~"
    result = tts(answer)
    save_and_upload_tts(result=result,filename=f"{msg_type}_{question_hash}")
    

if __name__ == "__main__":
    # 初始化
    init_tts()
    
    # 开辟新的线程 启动消费者 
    def start_consumer():
        try:
            rabbitmq_client.consume(callback=mq_callback)
        except Exception as e:
            print("消费者线程异常:", e)

    t = threading.Thread(target=start_consumer, daemon=True)
    t.start()
   
    while True:
        time.sleep(1)