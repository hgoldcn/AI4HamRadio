#!/usr/bin/env python3
import sys
import time
import sounddevice as sd
import sherpa_ncnn
import numpy as np
import ollama
import pyttsx3
import pygame  
import pygame.mixer  
import requests  
from io import BytesIO  
import text_speech_synthesis as tss
from datetime import datetime

last_audio_time = time.time()
current_result = ""
# 等候时间，当间隔两秒没有输入时，开始回复用户信息
output_delay = 2  # 秒
volume_threshold = 0.01  # 音频强度阈值（根据设备调整）

content = '你是一个非常好的智能助理，一位优秀的业余无线电爱好者。'
content += '请根据用户的输入回复，并遵循一下原则：'
content += '1、你的业余无线电台呼号为 BX1ABC；' # 请将呼号BX1ABC替换成自己的呼号
content += '2、你的地址是：北京市海淀区，梅登海德网格码是：OM89dw；'
content += '3、你用的设备是手持对讲机，型号为 FT1XD；'
content += '4、你的设备的输出功率是5w；'
content += '5、你设备等天线是定制的IOO拇指天线；'
content += '5、如果对方是业务无线电爱好者，你回复和交流的话题可以涉及到业余无线电的设备，发射功率，天线，信号报告，位置等内容；'
content += '6、Q简语是由国际电信联盟ITU规定和颁布的通信用短语，每个简语都由3个字母组成，第1个字母为Q，代表一个完整的意思。\
            这些Q简语可以单独使用，能够更加简练地表达自己的意思，从而提高通信效率。此外，还有一些英文缩写的缩语，常用的Q简语有：\
            QTH：表示询问位置。 \
            CQ：用于呼叫其他无线电操作员，表示欢迎他们回应你的呼叫。\
            QRZ：用于询问谁正在呼叫你，等待其他操作员回应。\
            Copy：表示接收到对方的通信内容。\
            QSL：确认接收到对方的信息，并请求交换通信确认卡片（QSL卡）。\
            73：表示美好的祝愿，是业余无线电爱好者交流后的结束语。在交流过程中不能使用，只有当对方说出73后，才能回复73。\
            Break：用于打断当前通信，表示有紧急消息或者重要事项需要传达。\
            SOS：国际求救信号，用于紧急情况。\
            TNX：表示感谢。\
            ICOM：代表某品牌无线电台制造商。'
content += '7、当用户输入类似这样的内容：CQ，CQ，CQ，这里是BA1ABC，\
            开始出现了CQ，或者连续的几个CQ，表示他在呼叫网络上的电台，你可以直接回复他， \
            后面跟着的5位或6位字母和数字的组合，如这里的BA1ABC，为业余无线电台的呼号。 \
            前面一个是用户的业余无线电台的呼号，后面一个是自己的业余无线电台呼号。 \
            回复如下：BA1ABC，这里是BX1ABC。'  # 请将呼号BX1ABC替换成自己的呼号
content += '8、当用户询问QTH时，你就回复你的位置信息：北京市海淀区， \
            必要的时候可以加上梅登海德网格码：OM89dw， \
            当用户询问你的QTH时，你也应该询问他的QTH，询问位置时直接用Q简语代替位置。'
content += '9、除了以上内容外，根据用户的输入进行正常回复'

messages = [
    {
        "role": "system",
        "content": content
    }
]

def create_recognizer():
    # 请修改 sherpa-ncnn 模型的目录
    recognizer = sherpa_ncnn.Recognizer(
        tokens="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt",
        encoder_param="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="sherpa-ncnn-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    return recognizer

# 直接利用 pyttsx3 库将文本转换成语音
def text2voice(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# 播放网络上的音频
def play_audio_from_url(url):  
    # 初始化pygame的mixer模块  
    pygame.mixer.init()  
       
    # 使用requests获取音频内容  
    response = requests.get(url)  
    audio_stream = BytesIO(response.content)  
       
    # 加载音频文件，注意pygame的mixer.Sound不支持直接从网络URL加载  
    # 因此我们需要先将音频内容下载到BytesIO对象中，然后从这里加载  
    sound = pygame.mixer.Sound(file=audio_stream)  
       
    # 播放音频  
    # pygame.mixer.Sound.play()  
    sound.play()
       
    # 保持程序运行直到音频播放完毕  
    while pygame.mixer.get_busy():  
        pygame.time.Clock().tick(10) 

def chat_with_ollama(messages):
        """与大语言模型交互"""
        output = ollama.chat(
            model="deepseek-r1:7b",     # Ollama里的大语言模型，如 deepseek-r1:7b 或 qwen2.5:latest
            messages=messages
        )
        model_output = output['message']['content']

        if model_output == '':
            model_output = '抱歉，我无法理解您的问题。'
        return(model_output)

def chat_with_ai(data):
    qustion = [{
        'role': 'user',
        'content': data
    }],
    messages.append(qustion[0][0])
    # 调用模型
    response = chat_with_ollama(messages)
    messages.append({
        'role': 'assistant',
        'content': response
    })
    now = datetime.now()
    crurrent_datetime_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    print(f"\n人工智能: {crurrent_datetime_str}, { response }")
    # text2voice(response)
    # 调用讯飞讯的文本转语音
    tss.main(response)
def main():
    global last_audio_time, current_result
    recognizer = create_recognizer()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.05 * sample_rate)  # 缩短读取间隔（0.05秒）

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            # 检测音频强度
            volume = np.abs(samples).max()
            if volume > volume_threshold:
                # 更新时间戳和处理音频
                last_audio_time = time.time()
                recognizer.accept_waveform(sample_rate, samples)
                current_result = recognizer.text.strip()
            else:
                # 喂入静默数据并更新结果
                recognizer.accept_waveform(sample_rate, samples)
                current_result = recognizer.text.strip()
            
            # 检查是否满足输出条件
            current_time = time.time()
            if (current_time - last_audio_time > output_delay) and current_result:
                # 等待0.5秒确保处理完成
                time.sleep(0.5)
                final_result = recognizer.text.strip()
                if final_result:
                    if final_result.lower() in ["exit", "quit", "stop", "baibai", '退出', '再见', '拜拜' ]:
                        sExit = '正在退出'
                        text2voice(sExit)
                        print(f"\n{ sExit } ……\n")
                        break
                    now = datetime.now()
                    crurrent_datetime_str = now.strftime('%Y-%m-%d_%H-%M-%S')
                    print(f"\n电    台: {crurrent_datetime_str}, {final_result}")
                    
                    chat_with_ai(final_result)

                recognizer.reset()
                current_result = ""
                last_audio_time = current_time

if __name__ == "__main__":
    devices = sd.query_devices()
    default_input_device_idx = sd.default.device[0]
    print(f'使用设备: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到终止信号，程序退出")
