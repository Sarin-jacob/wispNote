import io
import math
import time
import wave
import queue
import threading
import concurrent.futures
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
TARGET_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT_SEC = 0.8
MIN_SPEECH_DURATION_SEC = 0.5

# Diarization Tuning
MATCH_THRESHOLD = 0.38          # Above this = definite match
NEW_SPEAKER_THRESHOLD = 0.28    # Must be strictly lower than this to create Unknown_N
MIN_SEC_FOR_NEW_PROFILE = 1.2   # Audio must be at least this long to mint a new profile

# Chunking & Overlap Limits
SOFT_LIMIT_SEC = 14.0
HARD_LIMIT_SEC = 18.0
OVERLAP_SEC = 1.0
MICRO_SILENCE_TIMEOUT = 0.3

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
        self.enrolled_speakers = {}  # {name: torch.Tensor}
        self.unknown_count = 0
        self.last_speaker = "Unknown"

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
        current_duration = len(audio_float32) / TARGET_SAMPLE_RATE
        current_embedding = self.compute_embedding(audio_float32)

        # Baseline: If no profiles exist at all
        if not self.enrolled_speakers:
            self.unknown_count += 1
            new_name = f"Unknown_{self.unknown_count}"
            self.enrolled_speakers[new_name] = current_embedding
            self.last_speaker = new_name
            return new_name, 1.0

        best_name = "Unknown"
        highest_sim = -1.0

        # Compare against all known and auto-enrolled profiles
        for name, enrolled_emb in self.enrolled_speakers.items():
            similarity = torch.dot(current_embedding, enrolled_emb).item()
            if similarity > highest_sim:
                highest_sim = similarity
                best_name = name

        # 1. Strong Match: Update moving average centroid
        if highest_sim >= MATCH_THRESHOLD:
            # 80% existing baseline + 20% fresh utterance
            updated_emb = 0.80 * self.enrolled_speakers[best_name] + 0.20 * current_embedding
            self.enrolled_speakers[best_name] = updated_emb / torch.norm(updated_emb)
            self.last_speaker = best_name
            return best_name, highest_sim

        # 2. Ambiguous Region (0.28 <= sim < 0.38): Attribute to best match or last speaker
        if highest_sim >= NEW_SPEAKER_THRESHOLD or current_duration < MIN_SEC_FOR_NEW_PROFILE:
            return best_name, highest_sim

        # 3. Definite Divergence (< 0.28 and >= 1.2s of audio): Mint a new Unknown profile
        self.unknown_count += 1
        new_name = f"Unknown_{self.unknown_count}"
        self.enrolled_speakers[new_name] = current_embedding
        self.last_speaker = new_name
        return new_name, highest_sim

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
            resp = requests.post(self.api_url, files=files, data=data, timeout=30)
            resp.raise_for_status()
            response_data = resp.json()
            if isinstance(response_data, dict):
                return response_data.get("text", "").strip()
            return ""
        except Exception:
            return ""

# ==========================================
# 3. WASAPI / MIC DUAL AUDIO CAPTURER
# ==========================================
class AudioCapturer:
    def __init__(self, output_queue: queue.Queue, input_type: str = "both"):
        self.queue = output_queue
        self.input_type = input_type
        self.running = False
        self.p = pyaudio.PyAudio()

        self.sys_queue = queue.Queue()
        self.mic_queue = queue.Queue()
        self.sys_buffer, self.mic_buffer = [], []
        self.sys_device_index, self.mic_device_index = None, None

        self._find_devices()

    def _find_devices(self):
        if self.input_type in ["system", "both"]:
            try:
                wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                for loopback in self.p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        self.sys_device_index = loopback["index"]
                        break
                if self.sys_device_index is not None:
                    sys_info = self.p.get_device_info_by_index(self.sys_device_index)
                    self.sys_channels = int(sys_info["maxInputChannels"]) or 1
                    self.sys_rate = int(sys_info["defaultSampleRate"])
                    gcd = math.gcd(TARGET_SAMPLE_RATE, self.sys_rate)
                    self.sys_up, self.sys_down = TARGET_SAMPLE_RATE // gcd, self.sys_rate // gcd
            except OSError:
                print("[WARNING] WASAPI Loopback not found.")

        if self.input_type in ["mic", "both"]:
            try:
                self.mic_device_index = self.p.get_default_input_device_info()["index"]
                mic_info = self.p.get_device_info_by_index(self.mic_device_index)
                self.mic_channels = int(mic_info["maxInputChannels"]) or 1
                self.mic_rate = int(mic_info["defaultSampleRate"])
                gcd = math.gcd(TARGET_SAMPLE_RATE, self.mic_rate)
                self.mic_up, self.mic_down = TARGET_SAMPLE_RATE // gcd, self.mic_rate // gcd
            except OSError:
                print("[WARNING] Default microphone not found.")

    def _sys_callback(self, in_data, frame_count, time_info, status):
        raw = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.sys_channels)
        mono = raw.mean(axis=1).astype(np.float32) / 32768.0
        resampled = scipy.signal.resample_poly(mono, self.sys_up, self.sys_down)
        self.sys_buffer.extend(resampled)
        while len(self.sys_buffer) >= VAD_FRAME_SIZE:
            self.sys_queue.put(np.array(self.sys_buffer[:VAD_FRAME_SIZE]))
            self.sys_buffer = self.sys_buffer[VAD_FRAME_SIZE:]
        return (None, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        raw = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self.mic_channels)
        mono = raw.mean(axis=1).astype(np.float32) / 32768.0
        resampled = scipy.signal.resample_poly(mono, self.mic_up, self.mic_down)
        self.mic_buffer.extend(resampled)
        while len(self.mic_buffer) >= VAD_FRAME_SIZE:
            self.mic_queue.put(np.array(self.mic_buffer[:VAD_FRAME_SIZE]))
            self.mic_buffer = self.mic_buffer[VAD_FRAME_SIZE:]
        return (None, pyaudio.paContinue)

    def _mixer_thread(self):
        while self.running:
            sys_chunk = np.zeros(VAD_FRAME_SIZE, dtype=np.float32)
            mic_chunk = np.zeros(VAD_FRAME_SIZE, dtype=np.float32)
            got_audio = False

            if self.input_type in ["system", "both"]:
                try:
                    sys_chunk = self.sys_queue.get(timeout=0.01)
                    got_audio = True
                except queue.Empty: pass

            if self.input_type in ["mic", "both"]:
                try:
                    mic_chunk = self.mic_queue.get(timeout=0.01)
                    got_audio = True
                except queue.Empty: pass

            if got_audio:
                if self.input_type == "both":
                    mixed = np.clip(sys_chunk + mic_chunk, -1.0, 1.0)
                    self.queue.put(mixed)
                elif self.input_type == "system":
                    self.queue.put(sys_chunk)
                elif self.input_type == "mic":
                    self.queue.put(mic_chunk)
            else:
                time.sleep(0.01)

    def start(self):
        if self.running: return
        self.running = True
        self.streams = []

        if self.input_type in ["system", "both"] and self.sys_device_index is not None:
            s = self.p.open(format=pyaudio.paInt16, channels=self.sys_channels, rate=self.sys_rate,
                            input=True, frames_per_buffer=1024, input_device_index=self.sys_device_index,
                            stream_callback=self._sys_callback)
            s.start_stream()
            self.streams.append(s)

        if self.input_type in ["mic", "both"] and self.mic_device_index is not None:
            s = self.p.open(format=pyaudio.paInt16, channels=self.mic_channels, rate=self.mic_rate,
                            input=True, frames_per_buffer=1024, input_device_index=self.mic_device_index,
                            stream_callback=self._mic_callback)
            s.start_stream()
            self.streams.append(s)

        self.mix_thread = threading.Thread(target=self._mixer_thread, daemon=True)
        self.mix_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'streams'):
            for s in self.streams:
                try:
                    s.stop_stream()
                    s.close()
                except Exception: pass
        self.p.terminate()

# ==========================================
# 4. WISP PIPELINE COORDINATOR
# ==========================================
class WispPipeline:
    def __init__(self, whisper_url: str = WHISPER_API_URL, input_type: str = "both"):
        self.whisper_url = whisper_url
        self.input_type = input_type
        self.audio_queue = queue.Queue()
        self.capturer = AudioCapturer(self.audio_queue, input_type=input_type)
        self.speaker_id = SpeakerIdentifier()
        self.whisper = WhisperClient(whisper_url)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.transcript_subscribers = []  # Callbacks for FastAPI WebSockets

        print("[INIT] Loading Silero VAD (CPU)...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False
        )
        self.vad_model.eval()
        self.is_active = False

    def subscribe(self, callback):
        self.transcript_subscribers.append(callback)

    def enroll_speaker(self, name: str, duration_sec: int = 5):
        was_active = self.is_active
        if not self.capturer.running:
            self.capturer.start()

        chunks, samples_collected = [], 0
        samples_needed = TARGET_SAMPLE_RATE * duration_sec

        while not self.audio_queue.empty():
            self.audio_queue.get()

        while samples_collected < samples_needed:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                chunks.append(chunk)
                samples_collected += len(chunk)
            except queue.Empty: pass

        audio_clip = np.concatenate(chunks)[:samples_needed]
        self.speaker_id.enroll(name, audio_clip)

        if not was_active:
            self.capturer.stop()

    def start_pipeline(self):
        if self.is_active: return
        self.is_active = True
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def stop_pipeline(self):
        self.is_active = False
        self.capturer.stop()

    def _run_loop(self):
        self.capturer.start()
        audio_scratchpad, speech_frames = [], []
        is_speaking, silence_frame_count = False, 0
        
        micro_silence_limit = int((MICRO_SILENCE_TIMEOUT * TARGET_SAMPLE_RATE) / VAD_FRAME_SIZE)
        full_silence_limit = int((SILENCE_TIMEOUT_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SIZE)

        while self.is_active:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                audio_scratchpad.extend(chunk)
            except queue.Empty: continue

            while len(audio_scratchpad) >= VAD_FRAME_SIZE:
                frame = np.array(audio_scratchpad[:VAD_FRAME_SIZE], dtype=np.float32)
                audio_scratchpad = audio_scratchpad[VAD_FRAME_SIZE:]

                tensor_frame = torch.from_numpy(frame)
                with torch.no_grad():
                    speech_prob = self.vad_model(tensor_frame, TARGET_SAMPLE_RATE).item()

                if speech_prob >= VAD_THRESHOLD:
                    if not is_speaking: is_speaking = True
                    silence_frame_count = 0
                    speech_frames.append(frame)
                else:
                    if is_speaking:
                        speech_frames.append(frame)
                        silence_frame_count += 1

                if is_speaking:
                    current_dur = (len(speech_frames) * VAD_FRAME_SIZE) / TARGET_SAMPLE_RATE
                    hit_natural_end = silence_frame_count >= full_silence_limit
                    hit_soft_limit = current_dur >= SOFT_LIMIT_SEC and silence_frame_count >= micro_silence_limit
                    hit_hard_limit = current_dur >= HARD_LIMIT_SEC

                    if hit_natural_end or hit_soft_limit or hit_hard_limit:
                        total_audio = np.concatenate(speech_frames)
                        if (len(total_audio) / TARGET_SAMPLE_RATE) >= MIN_SPEECH_DURATION_SEC:
                            self.executor.submit(self._dispatch_segment, total_audio)

                        if hit_natural_end:
                            speech_frames = []
                            is_speaking = False
                        else:
                            overlap_count = int((OVERLAP_SEC * TARGET_SAMPLE_RATE) / VAD_FRAME_SIZE)
                            speech_frames = speech_frames[-overlap_count:] if len(speech_frames) > overlap_count else []
                        silence_frame_count = 0

    def _dispatch_segment(self, audio_segment: np.ndarray):
        timestamp = time.strftime("%H:%M:%S")
        speaker_name, score = self.speaker_id.identify(audio_segment)
        text = self.whisper.transcribe(audio_segment)

        if text:
            payload = {
                "timestamp": timestamp,
                "speaker": speaker_name,
                "similarity": round(score, 2),
                "text": text
            }
            # Distribute payload to WebSockets and console
            print(f"{timestamp}  {speaker_name:<12} (sim: {score:.2f})  {text}")
            for sub in self.transcript_subscribers:
                try: sub(payload)
                except Exception: pass