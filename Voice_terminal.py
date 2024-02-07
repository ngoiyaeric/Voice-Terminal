from pathlib import Path
import gradio as gr
import time
import io
from pydub import AudioSegment
from pydub.playback import play
import whisper
import openai
from interpreter import interpreter
from dotenv import dotenv_values

# Load variables from .env into a dictionary
env_vars = dotenv_values(".env")

# Set your OpenAI API key
openai.api_key = env_vars.get("API_KEY")
interpreter.llm.api_key = openai.api_key
interpreter.auto_run = True

model = "whisper-1"
def transcribe(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

# Function to get the length of audio in seconds
def get_audio_length(audio_bytes):
    try:
        byte_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_mp3(byte_io)
        length_s = len(audio) / 1000.0
        return length_s
    except Exception as e:
        print(f"Error in get_audio_length: {e}")
        return 0

# Function to generate speech using OpenAI TTS
def speak(text):
    try:
        response = openai.Audio.create(
            engine="tts-1",
            voice="alloy",
            text=text
        )

        audio_length = get_audio_length(response['audio'])

        speech_file_path = Path(__file__).parent // "speech.mp3"
        with open(speech_file_path, "wb") as f:
            f.write(response['audio'])

        play(response['audio'], notebook=True)
        time.sleep(audio_length)

    except Exception as e:
        print(f"Error in speak: {e}")

last_sentence = ""

with gr.Blocks() as demo:

    chatbot = gr.Chatbot()
    audio_input = gr.Audio(source="Microphone", type="filepath")
    btn = gr.Button("Submit")

    def transcribe(audio):
      audio = whisper.load_audio(audio)
      audio = whisper.pad_or_trim(audio)
      mel = whisper.log_mel_spectrogram(audio).to(model.device)
      _, probs = model.detect_language(mel)
      options = whisper.DecodingOptions()
      result = whisper.decode(model, mel, options)
      return result.text

    def add_user_message(audio, history):
        user_message = transcribe(audio)
        return history + [[user_message, None]]

    def bot(history):
        global last_sentence

        user_message = history[-1][0]
        history[-1][1] = ""
        active_block_type = ""
        language = ""
        for chunk in interpreter.chat(user_message, stream=True, display=False):

            if chunk["type"] == "message" and "content" in chunk:
              if active_block_type != "message":
                active_block_type = "message"
              history[-1][1] += chunk["content"]

              last_sentence += chunk["content"]
              if any([punct in last_sentence for punct in ".?!\n"]):
                yield history
                speak(last_sentence)
                last_sentence = ""
              else:
                yield history

            # Code
            if chunk["type"] == "code" and "content" in chunk:
              if active_block_type != "code":
                active_block_type = "code"
                history[-1][1] += f"\n```{chunk['format']}"
              history[-1][1] += chunk["content"]
              yield history

            # Output
            if chunk["type"] == "confirmation":
              history[-1][1] += "\n```\n\n```text\n"
              yield history
            if chunk["type"] == "console":
              if chunk.get("format") == "output":
                if chunk["content"] == "KeyboardInterrupt":
                  break
                history[-1][1] += chunk["content"] + "\n"
                yield history
              if chunk.get("format") == "active_line" and chunk["content"] == None:
                # Active line will be none when we finish execution.
                # You could also detect this with "type": "console", "end": True.
                history[-1][1] = history[-1][1].strip()
                history[-1][1] += "\n```\n"
                yield history

        if last_sentence:
          speak(last_sentence)

    btn.click(add_user_message, [audio_input, chatbot], [chatbot]).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch(debug=True)
