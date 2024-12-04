import os
from google.cloud import speech
from pydub import AudioSegment
import wave

# Constants
RATE = 16000  # Sampling rate

# Step 1: Check Google Cloud credentials
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
else:
    # Step 2: Input the audio file
    input_file = input("Enter the path to your audio file: ").strip()
    if not os.path.exists(input_file):
        print("Error: File not found. Please provide a valid file path.")
    else:
        # Step 3: Convert audio to WAV format if needed
        wav_path = "temp_audio.wav"
        if not input_file.endswith(".wav"):
            print("Converting to WAV format...")
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_frame_rate(RATE)  # Ensure the correct sample rate
            audio = audio.set_channels(1)  # Ensure mono audio
            audio.export(wav_path, format="wav")
        else:
            wav_path = input_file

        # Step 4: Validate WAV format (sample rate and channel count)
        with wave.open(wav_path, 'rb') as wf:
            if wf.getframerate() != RATE:
                print(f"Invalid sample rate: {wf.getframerate()}, expected {RATE}.")
            elif wf.getnchannels() != 1:
                print(f"Invalid channel count: {wf.getnchannels()}, expected mono.")
            else:
                # Step 5: Initialize Google Cloud Speech client
                client = speech.SpeechClient()

                # Step 6: Read audio content
                with open(wav_path, "rb") as audio_file:
                    audio_content = audio_file.read()

                # Step 7: Configure the request
                audio = speech.RecognitionAudio(content=audio_content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE,
                    language_code="en-US",
                )

                # Step 8: Perform the transcription
                response = client.recognize(config=config, audio=audio)

                # Step 9: Print the transcription
                transcription = ""
                for result in response.results:
                    transcription += result.alternatives[0].transcript

                print("\nTranscription:")
                print(transcription)

        # Step 10: Clean up temporary file
        if wav_path == "temp_audio.wav" and os.path.exists(wav_path):
            os.remove(wav_path)
