import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pipeline import WispPipeline

app = FastAPI(title="WispNotes Real-Time Meeting Transcriber")
pipeline = WispPipeline()

# Connection manager for active WebSockets
active_websockets = set()
loop = None

def broadcast_transcript(payload: dict):
    """Callback triggered from the background audio worker thread."""
    if loop and active_websockets:
        asyncio.run_coroutine_threadsafe(_send_to_all(payload), loop)

async def _send_to_all(payload: dict):
    for ws in list(active_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            active_websockets.discard(ws)

pipeline.subscribe(broadcast_transcript)

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

# ==========================================
# REST API ENDPOINTS
# ==========================================
class EnrollRequest(BaseModel):
    name: str
    duration: int = 5

class ConfigRequest(BaseModel):
    input_type: str # "system", "mic", or "both"

@app.post("/api/start")
def start_pipeline():
    pipeline.start_pipeline()
    return {"status": "started", "input_type": pipeline.capturer.input_type}

@app.post("/api/stop")
def stop_pipeline():
    pipeline.stop_pipeline()
    return {"status": "stopped"}

@app.post("/api/config")
def config_source(req: ConfigRequest):
    pipeline.stop_pipeline()
    pipeline.input_type = req.input_type
    pipeline.capturer.input_type = req.input_type
    pipeline.capturer._find_devices()
    pipeline.start_pipeline()
    return {"status": "updated", "input_type": req.input_type}

@app.post("/api/enroll")
def enroll_speaker(req: EnrollRequest):
    pipeline.enroll_speaker(req.name, req.duration)
    return {"status": "enrolled", "name": req.name}

@app.get("/api/speakers")
def get_speakers():
    return {
        "enrolled": list(pipeline.speaker_id.enrolled_speakers.keys()),
        "total_unknowns": pipeline.speaker_id.unknown_count
    }

# ==========================================
# WEBSOCKET ENDPOINT
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)

# ==========================================
# SINGLE PAGE HTML/JS FRONTEND
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WispNotes - Live Transcription</title>
        <style>
            :root {
                --bg: #0f172a;
                --panel: #1e293b;
                --text: #f8fafc;
                --subtext: #94a3b8;
                --primary: #38bdf8;
                --accent: #22c55e;
                --danger: #ef4444;
                --border: #334155;
            }
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                display: flex;
                height: 100vh;
                overflow: hidden;
            }
            /* Sidebar Controls */
            #sidebar {
                width: 320px;
                background: var(--panel);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                padding: 24px;
                box-sizing: border-box;
                gap: 20px;
            }
            h1 { font-size: 20px; margin: 0; display: flex; align-items: center; gap: 8px; }
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 9999px;
                background: rgba(148, 163, 184, 0.2);
                color: var(--subtext);
            }
            .status-badge.live {
                background: rgba(34, 197, 94, 0.2);
                color: var(--accent);
            }
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: currentColor;
            }
            .section-title {
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: var(--subtext);
                font-weight: 700;
                margin-bottom: 8px;
            }
            .btn-group { display: flex; gap: 8px; }
            button {
                flex: 1;
                padding: 10px 14px;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: #273549;
                color: white;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            button:hover { background: #334155; }
            button.primary { background: #0284c7; border-color: #0284c7; }
            button.primary:hover { background: #0369a1; }
            button.danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border-color: var(--danger); }
            button.danger:hover { background: var(--danger); color: white; }
            
            input, select {
                width: 100%;
                padding: 9px 12px;
                border-radius: 6px;
                border: 1px solid var(--border);
                background: #0f172a;
                color: white;
                box-sizing: border-box;
                font-size: 14px;
            }
            #speaker-list {
                list-style: none;
                padding: 0;
                margin: 0;
                overflow-y: auto;
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
            #speaker-list li {
                background: #0f172a;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border: 1px solid var(--border);
            }
            /* Main Transcript Area */
            #main {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 30px;
                overflow: hidden;
            }
            #header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border);
                padding-bottom: 16px;
                margin-bottom: 20px;
            }
            #transcript-box {
                flex: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 16px;
                padding-right: 12px;
            }
            .message {
                display: flex;
                flex-direction: column;
                gap: 4px;
                background: rgba(30, 41, 59, 0.5);
                padding: 14px 18px;
                border-radius: 8px;
                border-left: 3px solid var(--primary);
            }
            .message.Unknown { border-left-color: #64748b; }
            .msg-header {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
            }
            .speaker-tag {
                font-weight: 700;
                color: var(--primary);
            }
            .message.Unknown .speaker-tag { color: #94a3b8; }
            .msg-meta {
                color: var(--subtext);
                font-size: 12px;
            }
            .msg-body {
                font-size: 15px;
                line-height: 1.5;
                color: #e2e8f0;
            }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <div>
                <h1>WispNotes</h1>
                <div style="margin-top: 10px;">
                    <div id="live-badge" class="status-badge">
                        <div class="status-dot"></div>
                        <span id="live-text">STOPPED</span>
                    </div>
                </div>
            </div>

            <div>
                <div class="section-title">Controls</div>
                <div class="btn-group">
                    <button id="btn-toggle" class="primary" onclick="toggleEngine()">Start Listening</button>
                </div>
            </div>

            <div>
                <div class="section-title">Audio Source</div>
                <select id="audio-source" onchange="changeAudioSource()">
                    <option value="both">Mix Both (System + Mic)</option>
                    <option value="system">System Audio Only (WASAPI)</option>
                    <option value="mic">Microphone Only</option>
                </select>
            </div>

            <div>
                <div class="section-title">Speaker Enrollment</div>
                <div style="display: flex; gap: 6px;">
                    <input type="text" id="speaker-name" placeholder="Participant name...">
                    <button onclick="enrollSpeaker()" style="flex: 0 0 75px;">Enroll</button>
                </div>
                <div id="enroll-status" style="font-size: 12px; color: var(--subtext); margin-top: 4px;"></div>
            </div>

            <div style="display: flex; flex-direction: column; flex: 1; overflow: hidden;">
                <div class="section-title">Active Roster</div>
                <ul id="speaker-list"></ul>
            </div>
        </div>

        <div id="main">
            <div id="header">
                <div>
                    <h2 style="margin: 0; font-size: 18px;">Live Meeting Transcript</h2>
                    <span style="font-size: 13px; color: var(--subtext);">Whisper-v3 Turbo (NPU) + SpeechBrain Diarization (CPU)</span>
                </div>
                <button onclick="clearTranscript()" style="flex: 0 0 auto; padding: 6px 12px; font-size: 12px;">Clear View</button>
            </div>

            <div id="transcript-box"></div>
        </div>

        <script>
            let isRunning = false;
            let ws;

            function initWebSocket() {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws`);

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    appendMessage(data);
                };

                ws.onclose = () => {
                    setTimeout(initWebSocket, 2000);
                };
            }

            function appendMessage(item) {
                const box = document.getElementById('transcript-box');
                const div = document.createElement('div');
                div.className = `message ${item.speaker.startsWith('Unknown') ? 'Unknown' : ''}`;
                
                div.innerHTML = `
                    <div class="msg-header">
                        <span class="speaker-tag">${item.speaker}</span>
                        <span class="msg-meta">${item.timestamp} • sim: ${item.similarity}</span>
                    </div>
                    <div class="msg-body">${item.text}</div>
                `;
                
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
                updateSpeakerRoster();
            }

            async function toggleEngine() {
                const btn = document.getElementById('btn-toggle');
                const badge = document.getElementById('live-badge');
                const badgeText = document.getElementById('live-text');

                if (!isRunning) {
                    await fetch('/api/start', { method: 'POST' });
                    isRunning = true;
                    btn.textContent = 'Stop Engine';
                    btn.className = 'danger';
                    badge.className = 'status-badge live';
                    badgeText.textContent = '● LIVE';
                } else {
                    await fetch('/api/stop', { method: 'POST' });
                    isRunning = false;
                    btn.textContent = 'Start Listening';
                    btn.className = 'primary';
                    badge.className = 'status-badge';
                    badgeText.textContent = 'STOPPED';
                }
            }

            async function changeAudioSource() {
                const source = document.getElementById('audio-source').value;
                await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input_type: source })
                });
            }

            async function enrollSpeaker() {
                const input = document.getElementById('speaker-name');
                const status = document.getElementById('enroll-status');
                const name = input.value.trim();
                if (!name) return;

                status.textContent = `Recording 5s for '${name}'...`;
                input.value = '';

                const resp = await fetch('/api/enroll', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, duration: 5 })
                });

                if (resp.ok) {
                    status.textContent = `Enrolled '${name}' successfully!`;
                    updateSpeakerRoster();
                    setTimeout(() => status.textContent = '', 4000);
                }
            }

            async function updateSpeakerRoster() {
                const resp = await fetch('/api/speakers');
                const data = await resp.json();
                const list = document.getElementById('speaker-list');
                list.innerHTML = '';

                data.enrolled.forEach(spk => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${spk}</span><span style="color: var(--subtext); font-size: 11px;">Tracked</span>`;
                    list.appendChild(li);
                });
            }

            function clearTranscript() {
                document.getElementById('transcript-box').innerHTML = '';
            }

            initWebSocket();
            updateSpeakerRoster();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    # Serves on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)