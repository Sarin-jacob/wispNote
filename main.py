import io
import wave
import requests
import pyaudiowpatch as pyaudio

# Updated to your specific FastFlowLLM port
API_URL = "http://localhost:52625/v1/audio/transcriptions"
CHUNK = 1024

def capture_and_transcribe():
    p = pyaudio.PyAudio()
    
    # 1. Find the default WASAPI loopback device
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        device_index = None
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                device_index = loopback["index"]
                break
    except OSError:
        print("WASAPI not found. Ensure you are on Windows.")
        return

    if device_index is None:
        print("Could not find a valid loopback device.")
        return

    # 2. Extract the NATIVE sample rate and channels to prevent crashes
    device_info = p.get_device_info_by_index(device_index)
    channels = device_info["maxInputChannels"]
    rate = int(device_info["defaultSampleRate"])
    
    print(f"Starting audio stream... (Device: {device_index} | {rate} Hz | {channels} Channels)")
    
    # 3. Open stream with native settings
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index)

    print("Recording 5 seconds of system audio...")
    frames = []
    # Calculate total chunks needed for 5 seconds based on native rate
    for _ in range(0, int(rate / CHUNK * 5)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # 4. Save to in-memory WAV buffer using native settings
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    
    wav_buffer.seek(0)

    # 5. Send to FastFlowLLM Whisper endpoint
    print("Sending to FastFlowLLM Whisper endpoint...")
    files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
    data = {'model': 'whisper-1'}
    
    try:
        response = requests.post(API_URL, files=files, data=data)
        response.raise_for_status()
        print("\nTranscription Result:")
        print(response.json().get('text', 'No text found.'))
    except Exception as e:
        print(f"\nAPI Error: {e}")

if __name__ == "__main__":
    capture_and_transcribe()