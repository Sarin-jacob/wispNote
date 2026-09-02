import io
import math
import time
import wave
import queue
import numpy as np
import scipy.signal
import torch
import requests
import concurrent.futures
import pyaudiowpatch as pyaudio
from speechbrain.inference.speaker import EncoderClassifier

# ==========================================
# CONFIGURATION
# ==========================================
WHISPER_API_URL = "http://localhost:52625/v1/audio/transcriptions"
TARGET_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT_SEC = 0.8
MIN_SPEECH_DURATION_SEC = 0.5
SIMILARITY_THRESHOLD = 0.50 # Lowered slightly for better matches in varying noise
MAX_SPEECH_DURATION_SEC = 18.0

# ==========================================
# 1. SPEAKER IDENTIFIER (CPU)
# ==========================================
class SpeakerIdentifier:
    def __init__(self):
        print("[INIT] Loading Speaker Recognition Model (CPU)...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
        self.enrolled_speakers = {}

    def compute_embedding(self, audio_float32: np.ndarray) -> torch.Tensor:
        wav_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(wav_tensor).squeeze().cpu()
            embedding = embedding / torch.norm(embedding)
        return embedding

    def enroll(self, name: str, audio_float32: np.ndarray):
        embedding = self.compute_embedding(audio_float32)
        self.enrolled_speakers[name] = embedding
        print(f"[ENROLLED] '{name}' registered successfully.")

    def identify(self, audio_float32: np.ndarray) -> tuple[str, float]:
        if not self.enrolled_speakers:
            return "Unknown", 0.0

        current_embedding = self.compute_embedding(audio_float32)
        best_name, highest_sim = "Unknown", -1.0

        for name, enrolled_emb in self.enrolled_speakers.items():
            similarity = torch.dot(current_embedding, enrolled_emb).item()
            if similarity > highest_sim:
                highest_sim, best_name = similarity, name

        if highest_sim < SIMILARITY_THRESHOLD:
            return "Unknown", highest_sim
        return best_name, highest_sim

# ==========================================
# 2. FASTFLOWLLM WHISPER CLIENT (NPU)
# ==========================================
class WhisperClient:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def transcribe(self, audio_float32: np.ndarray) -> str:
        pcm16 = (audio_float32 * 32767.0).clip(-32768, 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        wav_buffer.seek(0)

        files = {'file': ('audio.wav', wav_buffer, 'audio/wav')}
        data = {
            'model': 'whisper-1',
            'prompt': 'This is a conversation in Hinglish, mixing English and Hindi words. The speakers are saying:'
        }

        try:
            resp = requests.post(self.api_url, files=files, data=data, timeout=120)
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as e:
            return f"[Transcription Error: {e}]"

# ==========================================
# 3. WASAPI SYSTEM AUDIO CAPTURER
# ==========================================
class AudioCapturer:
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
        
        gcd = math.gcd(TARGET_SAMPLE_RATE, self.native_rate)
        self.resample_up = TARGET_SAMPLE_RATE // gcd
        self.resample_down = self.native_rate // gcd

    def _audio_callback(self, in_data, frame_count, time_info, status):
        raw_audio = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.native_channels)
        mono = raw_audio.mean(axis=1).astype(np.float32) / 32768.0
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
        
        # Thread pool to handle API requests without blocking the audio loop
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        print("[INIT] Loading Silero VAD (CPU)...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_model.eval()

    def enroll_speaker(self, name: str, duration_sec: int = 5):
        # (Keep your existing enroll_speaker code here...)
        print(f"\n--- Enrolling Speaker: '{name}' ---")
        print(f"Play {name}'s voice for {duration_sec} seconds (Ensure audio is actively playing!)...")
        chunks, samples_collected = [], 0
        samples_needed = TARGET_SAMPLE_RATE * duration_sec
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while samples_collected < samples_needed:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                chunks.append(chunk)
                samples_collected += len(chunk)
            except queue.Empty:
                pass 
        audio_clip = np.concatenate(chunks)[:samples_needed]
        self.speaker_id.enroll(name, audio_clip)

    def run(self):
        if not self.capturer.running:
            self.capturer.start()
            
        print("\n" + "="*50)
        print(" [WISP] PIPELINE ACTIVE: Transcribing system audio...")
        print("="*50 + "\n")

        audio_scratchpad, speech_frames = [], []
        is_speaking, silence_frame_count = False, 0
        silence_frame_limit = int((SILENCE_TIMEOUT_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SIZE)

        try:
            while True:
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                    audio_scratchpad.extend(chunk)
                except queue.Empty:
                    continue 

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
                        
                        # 1. FORCED CHUNKING: Check if speech exceeds maximum allowed duration
                        current_duration = (len(speech_frames) * VAD_FRAME_SIZE) / TARGET_SAMPLE_RATE
                        if current_duration >= MAX_SPEECH_DURATION_SEC:
                            total_audio = np.concatenate(speech_frames)
                            # Offload to background thread
                            self.executor.submit(self._dispatch_segment, total_audio)
                            is_speaking, speech_frames, silence_frame_count = False, [], 0
                            
                    else:
                        if is_speaking:
                            speech_frames.append(frame)
                            silence_frame_count += 1
                            
                            # 2. NATURAL SILENCE: End of sentence detected
                            if silence_frame_count >= silence_frame_limit:
                                total_audio = np.concatenate(speech_frames)
                                if (len(total_audio) / TARGET_SAMPLE_RATE) >= MIN_SPEECH_DURATION_SEC:
                                    # Offload to background thread
                                    self.executor.submit(self._dispatch_segment, total_audio)
                                is_speaking, speech_frames, silence_frame_count = False, [], 0

        except KeyboardInterrupt:
            print("\nStopping audio pipeline...")
            self.capturer.stop()
            self.executor.shutdown(wait=False)

    def _dispatch_segment(self, audio_segment: np.ndarray):
        """Runs in a background thread so the VAD loop never stops listening."""
        timestamp = time.strftime("%H:%M:%S")
        speaker_name, score = self.speaker_id.identify(audio_segment)
        text = self.whisper.transcribe(audio_segment)

        if text:
            print(f"{timestamp}  {speaker_name:<10} (sim: {score:.2f})  {text}")

# ==========================================
# ENTRY POINT & CLI TESTING
# ==========================================
if __name__ == "__main__":
    pipeline = WispPipeline(whisper_url=WHISPER_API_URL)

    print("\n--- Speaker Enrollment Setup ---")
    print("Type a name to enroll. Press ENTER with a blank name when done to start transcribing.")
    
    pipeline.capturer.start() 

    while True:
        name = input("\nEnter speaker name (or press Enter to start live transcription): ").strip()
        if not name:
            break
        pipeline.enroll_speaker(name, duration_sec=5)

    pipeline.run()