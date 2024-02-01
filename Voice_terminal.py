from pathlib import Path
import gradio as gr
import time
import openai
import io
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
import whisper

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'
 # Replace with the actual path

client = OpenAI()


def transcribe(audio):
    try:
        audio = whisper.load_audio(audio)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(client.device)
        _, probs = client.detect_language(mel)
        options = whisper.DecodingOptions()
        result = whisper.decode(client, mel, options)
        return result.text
    except Exception as e:
        print(f"Error in transcribe: {e}")
        return "Error in transcribing audio"


# Function to get the length of audio in seconds
def get_audio_length(audio_bytes):
    try:
        byte_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_mp3(byte_io)
        length_ms = len(audio)
        length_s = length_ms / 1000.0
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

        speech_file_path = Path(__file__).parent / "speech.mp3"
        with open(speech_file_path, "wb") as f:
            f.write(response['audio'])

        play(response['audio'], notebook=True)
        time.sleep(audio_length)

    except Exception as e:
        print(f"Error in speak: {e}")

last_sentence = ""

# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    audio_input = gr.inputs.Audio(source="microphone", type="filepath")
    btn = gr.Button("Submit")

    # Function to transcribe audio and handle errors during click event
    def transcribe_and_handle_errors(audio):
        user_message = transcribe(audio)
        return history + [[user_message, None]]

    # Chatbot logic
    def bot(history):
        global last_sentence

        user_message = history[-1][0]
        history[-1][1] = ""
        active_block_type = ""

        # Iterate through chatbot responses
        for chunk in interpreter.chat(user_message, stream=True, display=False):
            try:
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

                if chunk["type"] == "code" and "content" in chunk:
                    if active_block_type != "code":
                        active_block_type = "code"
                        history[-1][1] += f"\n```{chunk['format']}"
                    history[-1][1] += chunk["content"]
                    yield history

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
                        history[-1][1] = history[-1][1].strip()
                        history[-1][1] += "\n```\n"
                        yield history

            except Exception as e:
                print(f"Error in bot: {e}")

        if last_sentence:
            speak(last_sentence)

    # Event handler for button click
    btn.click(transcribe_and_handle_errors, [audio_input, chatbot], [chatbot]).then(
        bot, chatbot, chatbot
    )

# Queue the demo and launch the Gradio interface in debug mode
demo.queue()
demo.launch(debug=True)
