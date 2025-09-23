from webui import tts_fn
import utils
import os
from config import config
from infer import infer, latest_version, get_net_g, infer_multilang
from scipy.io.wavfile import write

def save_tts_result(result, filename="output.wav"):
    """
    将 tts_fn 返回值保存为 wav 文件
    :param result: tts_fn 的返回值, 形如 ("Success", (sr, audio_array))
    :param filename: 保存的文件名
    """
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
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    print("speakers:",speakers)
    
    text ="你好世界"
    speaker= "mxj"
    sdp_ratio= 0.5
    noise_scale= 0.6
    noise_scale_w= 0.9
    length_scale= 1
    language= "ZH"
    reference_audio= None
    emotion= "Happy"
    prompt_mode= "Text prompt"
    style_text= ""
    style_weight= 0.7
    
    
    result  = tts_fn(     
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
           