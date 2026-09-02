import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pipeline import WispPipeline

app = FastAPI(title="WispNotes Real-Time Meeting Transcriber")
pipeline = WispPipeline()

active_websockets = set()
loop = None

def broadcast_transcript(payload: dict):
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
    input_type: str 

class LanguageRequest(BaseModel):
    language: str 

class RenameRequest(BaseModel):
    old_name: str
    new_name: str

@app.post("/api/start")
def start_pipeline():
    pipeline.start_pipeline()
    return {"status": "started"}

@app.post("/api/stop")
def stop_pipeline():
    pipeline.stop_pipeline()
    return {"status": "stopped"}

@app.post("/api/config")
def config_source(req: ConfigRequest):
    pipeline.stop_pipeline()
    pipeline.input_type = req.input_type
    pipeline.capturer.input_type = req.input_type
    pipeline.start_pipeline()
    return {"status": "updated"}

@app.post("/api/language")
def set_language(req: LanguageRequest):
    pipeline.whisper.language = req.language
    return {"status": "updated", "language": req.language}

@app.post("/api/enroll")
def enroll_speaker(req: EnrollRequest):
    pipeline.enroll_speaker(req.name, req.duration)
    return {"status": "enrolled"}

@app.post("/api/rename")
def rename_speaker(req: RenameRequest):
    pipeline.speaker_id.rename_speaker(req.old_name, req.new_name)
    return {"status": "renamed"}

@app.post("/api/clear_speakers")
def clear_speakers():
    pipeline.speaker_id.clear_speakers()
    return {"status": "cleared"}

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
            :root { --bg: #0f172a; --panel: #1e293b; --text: #f8fafc; --subtext: #94a3b8; --primary: #38bdf8; --border: #334155; }
            body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
            #sidebar { width: 340px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 24px; gap: 20px; box-sizing: border-box; }
            h1 { font-size: 20px; margin: 0; }
            .section-title { font-size: 12px; text-transform: uppercase; color: var(--subtext); font-weight: 700; margin-bottom: 8px; }
            button, select, input { width: 100%; padding: 9px 12px; border-radius: 6px; border: 1px solid var(--border); background: #273549; color: white; cursor: pointer; box-sizing: border-box; }
            button:hover { background: #334155; }
            button.primary { background: #0284c7; border-color: #0284c7; font-weight: bold; }
            button.danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; border-color: #ef4444; }
            #speaker-list { list-style: none; padding: 0; margin: 0; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 6px; }
            #speaker-list li { background: var(--bg); padding: 8px 12px; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; border: 1px solid var(--border); cursor: pointer; }
            #speaker-list li:hover { border-color: var(--primary); }
            
            #main { flex: 1; display: flex; flex-direction: column; padding: 30px; overflow: hidden; }
            #header { display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }
            #transcript-box { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-right: 12px; }
            .message { display: flex; flex-direction: column; gap: 4px; background: rgba(30, 41, 59, 0.5); padding: 14px 18px; border-radius: 8px; border-left: 3px solid var(--primary); }
            .message.Unknown { border-left-color: #64748b; }
            .speaker-tag { font-weight: 700; color: var(--primary); }
            .msg-meta { color: var(--subtext); font-size: 12px; margin-left: 10px; }
            .msg-body { font-size: 15px; line-height: 1.5; color: #e2e8f0; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <div>
                <h1>WispNotes</h1>
                <span id="live-text" style="color: var(--subtext); font-size: 12px; font-weight: bold;">● STOPPED</span>
            </div>

            <div>
                <button id="btn-toggle" class="primary" onclick="toggleEngine()">Start Listening</button>
            </div>

            <div>
                <div class="section-title">Configuration</div>
                <select id="audio-source" onchange="changeAudioSource()" style="margin-bottom: 8px;">
                    <option value="both">Mix Both (System + Mic)</option>
                    <option value="system">System Audio Only (WASAPI)</option>
                    <option value="mic">Microphone Only</option>
                </select>
                <select id="lang-source" onchange="changeLanguage()">
                    <option value="hinglish">Language: Hinglish</option>
                    <option value="english">Language: English Only</option>
                    <option value="auto">Language: Auto-Detect</option>
                </select>
            </div>

            <div>
                <div class="section-title">Enroll Speaker</div>
                <div style="display: flex; gap: 6px;">
                    <input type="text" id="speaker-name" placeholder="Name...">
                    <button onclick="enrollSpeaker()" style="flex: 0 0 75px;">Enroll</button>
                </div>
                <div id="enroll-status" style="font-size: 12px; color: var(--primary); margin-top: 4px;"></div>
            </div>

            <div style="display: flex; flex-direction: column; flex: 1; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="section-title" style="margin: 0;">Roster (Click to Rename)</div>
                    <button onclick="clearSpeakers()" style="width: auto; padding: 2px 8px; font-size: 11px;">Clear All</button>
                </div>
                <ul id="speaker-list" style="margin-top: 8px;"></ul>
            </div>
        </div>

        <div id="main">
            <div id="header">
                <h2>Live Transcript</h2>
                <div style="display: flex; gap: 10px;">
                    <button onclick="downloadTranscript()">Download .TXT</button>
                    <button onclick="clearTranscript()">Clear View</button>
                </div>
            </div>
            <div id="transcript-box"></div>
        </div>

        <script>
            let isRunning = false;
            let expectedSeq = 0;
            const pendingChunks = {}; 
            let ws;

            function initWebSocket() {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws`);

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    // Sequence Buffering Logic to fix out-of-order chunks
                    pendingChunks[data.seq] = data;
                    
                    while (pendingChunks[expectedSeq]) {
                        displayMessage(pendingChunks[expectedSeq]);
                        delete pendingChunks[expectedSeq];
                        expectedSeq++;
                    }
                };
                ws.onclose = () => setTimeout(initWebSocket, 2000);
            }

            function displayMessage(item) {
                const box = document.getElementById('transcript-box');
                const div = document.createElement('div');
                div.className = `message ${item.speaker.startsWith('Unknown') ? 'Unknown' : ''}`;
                
                // Add hidden data attributes for the download extractor
                div.setAttribute('data-speaker', item.speaker);
                div.setAttribute('data-time', item.timestamp);
                div.setAttribute('data-text', item.text);

                div.innerHTML = `
                    <div>
                        <span class="speaker-tag">${item.speaker}</span>
                        <span class="msg-meta">${item.timestamp}</span>
                    </div>
                    <div class="msg-body">${item.text}</div>
                `;
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
                updateSpeakerRoster();
            }

            async function toggleEngine() {
                const btn = document.getElementById('btn-toggle');
                const badge = document.getElementById('live-text');
                if (!isRunning) {
                    await fetch('/api/start', { method: 'POST' });
                    isRunning = true;
                    expectedSeq = 0; // Reset sequencing on start
                    btn.textContent = 'Stop Engine'; btn.className = 'danger';
                    badge.textContent = '● LIVE'; badge.style.color = '#22c55e';
                } else {
                    await fetch('/api/stop', { method: 'POST' });
                    isRunning = false;
                    btn.textContent = 'Start Listening'; btn.className = 'primary';
                    badge.textContent = '● STOPPED'; badge.style.color = 'var(--subtext)';
                }
            }

            async function changeAudioSource() {
                const source = document.getElementById('audio-source').value;
                await fetch('/api/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({input_type: source}) });
            }

            async function changeLanguage() {
                const lang = document.getElementById('lang-source').value;
                await fetch('/api/language', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({language: lang}) });
            }

            async function enrollSpeaker() {
                const input = document.getElementById('speaker-name');
                const status = document.getElementById('enroll-status');
                if (!input.value.trim()) return;
                status.textContent = `Recording 5s for '${input.value}'...`;
                
                const resp = await fetch('/api/enroll', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name: input.value.trim(), duration: 5}) });
                if (resp.ok) {
                    status.textContent = `Enrolled successfully!`;
                    input.value = '';
                    updateSpeakerRoster();
                    setTimeout(() => status.textContent = '', 3000);
                }
            }

            async function clearSpeakers() {
                if(confirm("Are you sure you want to clear all enrolled speakers?")) {
                    await fetch('/api/clear_speakers', { method: 'POST' });
                    updateSpeakerRoster();
                }
            }

            async function updateSpeakerRoster() {
                const resp = await fetch('/api/speakers');
                const data = await resp.json();
                const list = document.getElementById('speaker-list');
                list.innerHTML = '';

                data.enrolled.forEach(spk => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${spk}</span> <span style="font-size: 10px;">✏️</span>`;
                    li.onclick = async () => {
                        const newName = prompt(`Rename ${spk} to:`);
                        if (newName && newName.trim() !== "") {
                            await fetch('/api/rename', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({old_name: spk, new_name: newName.trim()}) });
                            updateSpeakerRoster();
                        }
                    };
                    list.appendChild(li);
                });
            }

            function clearTranscript() {
                document.getElementById('transcript-box').innerHTML = '';
            }

            function downloadTranscript() {
                const msgs = document.querySelectorAll('.message');
                let textContent = "WispNotes Transcript\\n====================\\n\\n";
                
                msgs.forEach(msg => {
                    const speaker = msg.getAttribute('data-speaker');
                    const time = msg.getAttribute('data-time');
                    const text = msg.getAttribute('data-text');
                    textContent += `[${time}] ${speaker}:\\n${text}\\n\\n`;
                });

                const blob = new Blob([textContent], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `Meeting_Transcript_${new Date().toISOString().slice(0,10)}.txt`;
                a.click();
                URL.revokeObjectURL(url);
            }

            initWebSocket();
            updateSpeakerRoster();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)