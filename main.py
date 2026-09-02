import io
import math
import time
import wave
import threading
import queue
import numpy as np
import scipy.signal
import torch
import requests
import pyaudiowpatch as pyaudio
from speechbrain.inference.speaker import EncoderClassifier

# ==========================================
# CONFIGURATION
# ==========================================
WHISPER_API_URL = "http://localhost:52625/v1/audio/transcriptions"
TARGET_SAMPLE_RATE = 16000  # 16 kHz required by Whisper & VAD
VAD_FRAME_SIZE = 512        # 32ms frames at 16 kHz for Silero
VAD_THRESHOLD = 0.5         # Speech probability threshold
SILENCE_TIMEOUT_SEC = 0.8   # Silence duration to finalize an utterance
MIN_SPEECH_DURATION_SEC = 0.5 # Minimum speech segment to transcribe
SIMILARITY_THRESHOLD = 0.60 # Cosine similarity threshold for speaker match

# ==========================================
# 1. SPEAKER IDENTIFIER (CPU)
# ==========================================
class SpeakerIdentifier:
    """Extracts speaker embeddings and performs cosine similarity matching."""
    def __init__(self):
        print("[INIT] Loading Speaker Recognition Model (SpeechBrain ECAPA-TDNN on CPU)...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        self.enrolled_speakers = {}  # {name: torch.Tensor}

    def compute_embedding(self, audio_float32: np.ndarray) -> torch.Tensor:
        """Computes a normalized speaker embedding vector."""
        wav_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(wav_tensor)
            embedding = embedding.squeeze().cpu()
            # Normalize embedding
            embedding = embedding / torch.norm(embedding)
        return embedding

    def enroll(self, name: str, audio_float32: np.ndarray):
        """Registers a participant's voice embedding."""
        embedding = self.compute_embedding(audio_float32)
        self.enrolled_speakers[name] = embedding
        print(f"[ENROLLED] Speaker '{name}' registered successfully.")

    def identify(self, audio_float32: np.ndarray) -> tuple[str, float]:
        """Matches audio segment against enrolled embeddings using cosine similarity."""
        if not self.enrolled_speakers:
            return "Unknown", 0.0

        current_embedding = self.compute_embedding(audio_float32)
        best_name = "Unknown"
        highest_sim = -1.0

        for name, enrolled_emb in self.enrolled_speakers.items():
            similarity = torch.dot(current_embedding, enrolled_emb).item()
            if similarity > highest_sim:
                highest_sim = similarity
                best_name = name

        if highest_sim < SIMILARITY_THRESHOLD:
            return "Unknown", highest_sim
        return best_name, highest_sim

# ==========================================
# 2. FASTFLOWLLM WHISPER CLIENT (NPU)
# ==========================================
class WhisperClient:
    """Handles speech-to-text requests to the FastFlowLLM OpenAI-compatible endpoint."""
    def __init__(self, api_url: str):
        self.api_url = api_url

    def transcribe(self, audio_float32: np.ndarray) -> str:
        # Convert float32 [-1.0, 1.0] to 16-bit PCM WAV
        pcm16 = (audio_float32 * 32767.0).clip(-32768, 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        wav_buffer.seek(0)

        files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
        data = {'model': 'whisper-1'}

        try:
            resp = requests.post(self.api_url, files=files, data=data, timeout=10)
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as e:
            return f"[Transcription Error: {e}]"

# ==========================================
# 3. WASAPI SYSTEM AUDIO CAPTURER
# ==========================================
class AudioCapturer:
    """Continuous WASAPI loopback capture with dynamic 16 kHz resampling."""
    def __init__(self, output_queue: queue.Queue):
        self.queue = output_queue
        self.running = False
        self.p = pyaudio.PyAudio()
        self._find_device()

    def _find_device(self):
        wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        self.device_index = None
        for loopback in self.p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                self.device_index = loopback["index"]
                break
        
        if self.device_index is None:
            raise RuntimeError("No WASAPI loopback device found.")

        device_info = self.p.get_device_info_by_index(self.device_index)
        self.native_channels = device_info["maxInputChannels"]
        self.native_rate = int(device_info["defaultSampleRate"])
        
        # Calculate resampling factors
        gcd = math.gcd(TARGET_SAMPLE_RATE, self.native_rate)
        self.resample_up = TARGET_SAMPLE_RATE // gcd
        self.resample_down = self.native_rate // gcd

    def _audio_callback(self, in_data, frame_count, time_info, status):
        # Convert raw bytes to numpy array
        raw_audio = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.native_channels)
        
        # Convert to mono float32 [-1.0, 1.0]
        mono = raw_audio.mean(axis=1).astype(np.float32) / 32768.0
        
        # Resample to 16 kHz
        resampled = scipy.signal.resample_poly(mono, self.resample_up, self.resample_down)
        self.queue.put(resampled)
        return (None, pyaudio.paContinue)

    def start(self):
        self.running = True
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.native_channels,
            rate=self.native_rate,
            input=True,
            frames_per_buffer=1024,
            input_device_index=self.device_index,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# ==========================================
# 4. SILERO VAD & PIPELINE COORDINATOR
# ==========================================
class WispPipeline:
    def __init__(self, whisper_url: str):
        self.audio_queue = queue.Queue()
        self.capturer = AudioCapturer(self.audio_queue)
        self.speaker_id = SpeakerIdentifier()
        self.whisper = WhisperClient(whisper_url)

        print("[INIT] Loading Silero VAD (CPU)...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_model.eval()

    def enroll_speaker(self, name: str, duration_sec: int = 5):
        """Helper to enroll a speaker directly from current system output/audio."""
        print(f"\n--- Enrolling Speaker: '{name}' ---")
        print(f"Play {name}'s voice for {duration_sec} seconds...")
        
        chunks = []
        samples_needed = TARGET_SAMPLE_RATE * duration_sec
        samples_collected = 0

        # Drain old audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        while samples_collected < samples_needed:
            chunk = self.audio_queue.get()
            chunks.append(chunk)
            samples_collected += len(chunk)

        audio_clip = np.concatenate(chunks)[:samples_needed]
        self.speaker_id.enroll(name, audio_clip)

    def run(self):
        """Starts real-time transcription and speaker identification loop."""
        self.capturer.start()
        print("\n" + "="*50)
        print(" [WISP] PIPELINE ACTIVE: Transcribing system audio...")
        print("="*50 + "\n")

        audio_scratchpad = []
        speech_frames = []
        is_speaking = False
        silence_frame_count = 0
        silence_frame_limit = int((SILENCE_TIMEOUT_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SIZE)

        try:
            while True:
                # 1. Fetch raw resampled audio chunks
                chunk = self.audio_queue.get()
                audio_scratchpad.extend(chunk)

                # 2. Slice into 512-sample frames for Silero VAD
                while len(audio_scratchpad) >= VAD_FRAME_SIZE:
                    frame = np.array(audio_scratchpad[:VAD_FRAME_SIZE], dtype=np.float32)
                    audio_scratchpad = audio_scratchpad[VAD_FRAME_SIZE:]

                    tensor_frame = torch.from_numpy(frame)
                    with torch.no_grad():
                        speech_prob = self.vad_model(tensor_frame, TARGET_SAMPLE_RATE).item()

                    if speech_prob >= VAD_THRESHOLD:
                        if not is_speaking:
                            is_speaking = True
                        silence_frame_count = 0
                        speech_frames.append(frame)
                    else:
                        if is_speaking:
                            speech_frames.append(frame)
                            silence_frame_count += 1

                            # 3. Silence threshold reached: process complete utterance
                            if silence_frame_count >= silence_frame_limit:
                                total_audio = np.concatenate(speech_frames)
                                duration = len(total_audio) / TARGET_SAMPLE_RATE

                                if duration >= MIN_SPEECH_DURATION_SEC:
                                    self._dispatch_segment(total_audio)

                                # Reset VAD state
                                is_speaking = False
                                speech_frames = []
                                silence_frame_count = 0

        except KeyboardInterrupt:
            print("\nStopping audio pipeline...")
            self.capturer.stop()

    def _dispatch_segment(self, audio_segment: np.ndarray):
        """Sends segment to FastFlowLLM (NPU) and Speaker ID (CPU)."""
        timestamp = time.strftime("%H:%M:%S")

        # Step A: Speaker Identification (CPU)
        speaker_name, score = self.speaker_id.identify(audio_segment)

        # Step B: Speech-to-Text (FastFlowLLM / NPU)
        text = self.whisper.transcribe(audio_segment)

        if text:
            # Output matching the WispNotes transcript spec
            print(f"{timestamp}  {speaker_name:<10} (sim: {score:.2f})  {text}")

# ==========================================
# ENTRY POINT & CLI TESTING
# ==========================================
if __name__ == "__main__":
    pipeline = WispPipeline(whisper_url=WHISPER_API_URL)

    # Optional quick enrollment prompt before starting live capture
    print("\n--- Speaker Enrollment Setup ---")
    enroll_choice = input("Do you want to enroll a participant now? (y/n): ").strip().lower()
    if enroll_choice == 'y':
        pipeline.capturer.start() # Start capture temporarily for enrollment
        name = input("Enter speaker name (e.g., Sarin, Alan): ").strip()
        pipeline.enroll_speaker(name, duration_sec=5)
        pipeline.capturer.stop()

    # Launch the live engine
    pipeline.run()