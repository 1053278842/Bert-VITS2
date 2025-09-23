from webui import tts_fn
import utils
import os
import numpy as np
from config import config
from infer import latest_version, get_net_g
from scipy.io.wavfile import write
import webui   # 注意：额外导入 webui 模块本身，用于注入全局变量

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

    text = "你好世界"
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

    save_tts_result(result, "hello.wav")
