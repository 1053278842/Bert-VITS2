from webui import tts_fn
import utils
import os
import io
import numpy as np
from config import config
from infer import latest_version, get_net_g
from scipy.io.wavfile import write
import datetime
import webui   # 注意：额外导入 webui 模块本身，用于注入全局变量
import requests
import soundfile as sf
from datetime import datetime

url = "http://121.36.251.16:7999/api/upload"

def save_and_upload_tts(result, folder="output"):
    """
    保存 tts_fn 结果到本地 wav 文件 → 上传到云端 → 删除本地文件
    :param result: tts_fn 返回值, 形如 ("Success", (sr, audio_array))
    :param folder: 临时保存的文件夹
    """
    status, audio_data = result
    if status != "Success":
        raise ValueError(f"TTS 失败: {status}")

    sr, audio_array = audio_data

    # 确保保存目录存在
    os.makedirs(folder, exist_ok=True)

    # 文件名 audio_20250923_153000.wav
    filename = f"audio.wav"
    # filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
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
        
def save_tts_result(result, filename="output.wav"):
    status, audio_data = result
    if status != "Success":
        raise ValueError(f"TTS 失败: {status}")

    sr, audio_array = audio_data
    write(filename, sr, audio_array.astype(np.int16))
    print(f"已保存音频到 {filename}")


if __name__ == "__main__":
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

    text = "你好，主人，我是墨小菊~"
    speaker = "mxj"
    sdp_ratio = 0.5
    noise_scale = 0.6
    noise_scale_w = 0.9
    length_scale = 1
    language = "ZH"
    reference_audio = None
    emotion = "Happy"
    prompt_mode = "Text prompt"
    style_text = ""
    style_weight = 0.7

    result = tts_fn(
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

    save_and_upload_tts(result=result)
