from webui import tts_fn

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
           