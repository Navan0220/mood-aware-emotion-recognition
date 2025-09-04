import pyaudio
import wave
import os
from datetime import datetime

# Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 15
OUTPUT_FOLDER = "../../data/raw/audio"

# Create folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Generate unique filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{timestamp}.wav"
output_path = os.path.join(OUTPUT_FOLDER, filename)

# Initialize PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=1024)

print("Recording... Speak now!")
frames = []

# Record audio
for i in range(0, int(RATE / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("Finished recording!")

# Stop and close stream
stream.stop_stream()
stream.close()
p.terminate()

# Save recording
wf = wave.open(output_path, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"✅ Audio saved as: {output_path}")