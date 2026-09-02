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
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            :root { 
                --bg: #0b1120; 
                --panel: #1e293b; 
                --text: #f8fafc; 
                --subtext: #94a3b8; 
                --primary: #3b82f6; 
                --primary-hover: #2563eb;
                --border: #334155;
                --danger: #ef4444;
                --success: #10b981;
            }
            
            body { 
                margin: 0; 
                font-family: 'Inter', -apple-system, sans-serif; 
                background: var(--bg); 
                color: var(--text); 
                display: flex; 
                height: 100vh; 
                overflow: hidden; 
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #64748b; }

            /* Sidebar */
            #sidebar { 
                width: 360px; 
                background: var(--panel); 
                border-right: 1px solid var(--border); 
                display: flex; 
                flex-direction: column; 
                padding: 24px; 
                gap: 24px; 
                box-sizing: border-box; 
                box-shadow: 4px 0 15px rgba(0,0,0,0.2);
                z-index: 10;
            }
            
            .header-container { display: flex; justify-content: space-between; align-items: center; }
            h1 { font-size: 22px; margin: 0; font-weight: 700; letter-spacing: -0.5px;}
            
            .status-badge {
                display: flex; align-items: center; gap: 6px;
                padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
                background: rgba(148, 163, 184, 0.1); color: var(--subtext); transition: all 0.3s ease;
            }
            .status-badge.live { background: rgba(16, 185, 129, 0.15); color: var(--success); }
            .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
            
            .section-title { font-size: 11px; text-transform: uppercase; color: var(--subtext); font-weight: 700; letter-spacing: 0.5px; margin-bottom: 10px; }
            
            button, select, input { 
                width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); 
                background: rgba(15, 23, 42, 0.6); color: white; font-family: 'Inter', sans-serif; font-size: 14px;
                outline: none; transition: all 0.2s;
            }
            input:focus, select:focus { border-color: var(--primary); }
            
            button { background: #273549; cursor: pointer; font-weight: 500; }
            button:hover { background: #334155; }
            button.primary { background: var(--primary); border-color: var(--primary); font-weight: 600; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
            button.primary:hover { background: var(--primary-hover); }
            button.danger { background: rgba(239, 68, 68, 0.15); color: var(--danger); border-color: rgba(239, 68, 68, 0.3); }
            button.danger:hover { background: var(--danger); color: white; }
            
            /* Roster List */
            #speaker-list { list-style: none; padding: 0; margin: 0; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 8px; }
            #speaker-list li { 
                background: rgba(15, 23, 42, 0.4); padding: 10px 14px; border-radius: 8px; font-size: 14px; 
                display: flex; justify-content: space-between; align-items: center; border: 1px solid transparent; 
                cursor: pointer; transition: all 0.2s;
            }
            #speaker-list li:hover { border-color: var(--primary); background: rgba(15, 23, 42, 0.8); transform: translateY(-1px);}
            
            /* Main Chat Area */
            #main { flex: 1; display: flex; flex-direction: column; background: var(--bg); position: relative; }
            
            #header { 
                display: flex; justify-content: space-between; align-items: center; 
                padding: 24px 40px; border-bottom: 1px solid var(--border); background: rgba(11, 17, 32, 0.8);
                backdrop-filter: blur(10px); z-index: 5;
            }
            #header h2 { margin: 0; font-size: 18px; font-weight: 600; }
            .header-buttons { display: flex; gap: 12px; }
            .header-buttons button { width: auto; padding: 8px 16px; font-size: 13px; border-radius: 20px; }
            
            #transcript-box { 
                flex: 1; overflow-y: auto; display: flex; flex-direction: column; 
                gap: 20px; padding: 30px 40px; scroll-behavior: smooth;
            }
            
            /* Message Bubbles */
            .message-wrapper { display: flex; gap: 16px; animation: fadeIn 0.3s ease forwards; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            .avatar {
                width: 40px; height: 40px; border-radius: 50%; background: var(--primary); 
                display: flex; align-items: center; justify-content: center; font-weight: 600; 
                font-size: 16px; color: white; flex-shrink: 0; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            }
            .message-wrapper.Unknown .avatar { background: #475569; }
            
            .message-content {
                display: flex; flex-direction: column; gap: 6px; max-width: 85%;
                background: var(--panel); padding: 16px 20px; border-radius: 0 16px 16px 16px;
                border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .msg-header { display: flex; align-items: baseline; gap: 12px; }
            .speaker-tag { font-weight: 600; color: var(--primary); font-size: 14px; }
            .message-wrapper.Unknown .speaker-tag { color: #94a3b8; }
            .msg-meta { color: #64748b; font-size: 11px; font-weight: 500; }
            
            .msg-body { font-size: 15px; line-height: 1.6; color: #e2e8f0; }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <div class="header-container">
                <h1>WispNotes</h1>
                <div id="live-badge" class="status-badge">
                    <div class="dot"></div>
                    <span id="live-text">STOPPED</span>
                </div>
            </div>

            <div>
                <button id="btn-toggle" class="primary" onclick="toggleEngine()">Start Listening</button>
            </div>

            <div>
                <div class="section-title">Configuration</div>
                <select id="audio-source" onchange="changeAudioSource()" style="margin-bottom: 12px;">
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
                <div style="display: flex; gap: 8px;">
                    <input type="text" id="speaker-name" placeholder="Enter name...">
                    <button onclick="enrollSpeaker()" style="flex: 0 0 80px;">Enroll</button>
                </div>
                <div id="enroll-status" style="font-size: 12px; color: var(--success); margin-top: 6px; font-weight: 500; height: 14px;"></div>
            </div>

            <div style="display: flex; flex-direction: column; flex: 1; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div class="section-title" style="margin: 0;">Roster (Click to Rename)</div>
                    <button onclick="clearSpeakers()" style="width: auto; padding: 4px 10px; font-size: 11px; border-radius: 12px;">Clear All</button>
                </div>
                <ul id="speaker-list"></ul>
            </div>
        </div>

        <div id="main">
            <div id="header">
                <h2>Live Meeting Transcript</h2>
                <div class="header-buttons">
                    <button onclick="downloadTranscript()" style="background: rgba(255,255,255,0.1); border:none;">Download .TXT</button>
                    <button onclick="clearTranscript()" style="background: rgba(255,255,255,0.1); border:none;">Clear View</button>
                </div>
            </div>
            <div id="transcript-box">
                <!-- Transcript messages will inject here -->
            </div>
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
                const isUnknown = item.speaker.startsWith('Unknown');
                
                div.className = `message-wrapper ${isUnknown ? 'Unknown' : ''}`;
                div.setAttribute('data-speaker', item.speaker);
                div.setAttribute('data-time', item.timestamp);
                div.setAttribute('data-text', item.text);

                const initial = isUnknown ? '?' : item.speaker.charAt(0).toUpperCase();

                div.innerHTML = `
                    <div class="avatar">${initial}</div>
                    <div class="message-content">
                        <div class="msg-header">
                            <span class="speaker-tag">${item.speaker}</span>
                            <span class="msg-meta">${item.timestamp}</span>
                        </div>
                        <div class="msg-body">${item.text}</div>
                    </div>
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
                    expectedSeq = 0; 
                    btn.textContent = 'Stop Engine'; btn.className = 'danger';
                    badge.className = 'status-badge live'; badgeText.textContent = 'LIVE';
                } else {
                    await fetch('/api/stop', { method: 'POST' });
                    isRunning = false;
                    btn.textContent = 'Start Listening'; btn.className = 'primary';
                    badge.className = 'status-badge'; badgeText.textContent = 'STOPPED';
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

            // --- RETROACTIVE RENAME LOGIC ---
            async function updateSpeakerRoster() {
                const resp = await fetch('/api/speakers');
                const data = await resp.json();
                const list = document.getElementById('speaker-list');
                list.innerHTML = '';

                data.enrolled.forEach(spk => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${spk}</span> <span style="font-size: 12px; opacity: 0.6;">✎ Edit</span>`;
                    
                    li.onclick = async () => {
                        const newName = prompt(`Rename ${spk} to:`);
                        if (newName && newName.trim() !== "") {
                            const formattedNewName = newName.trim();
                            const renameResp = await fetch('/api/rename', { 
                                method: 'POST', 
                                headers: {'Content-Type':'application/json'}, 
                                body: JSON.stringify({old_name: spk, new_name: formattedNewName}) 
                            });
                            
                            if (renameResp.ok) {
                                // 1. Update the UI DOM retroactively
                                const messages = document.querySelectorAll('.message-wrapper');
                                messages.forEach(msg => {
                                    if (msg.getAttribute('data-speaker') === spk) {
                                        // Update tracking attribute
                                        msg.setAttribute('data-speaker', formattedNewName);
                                        
                                        // Update UI Text
                                        msg.querySelector('.speaker-tag').textContent = formattedNewName;
                                        
                                        // Update Avatar Initial
                                        msg.querySelector('.avatar').textContent = formattedNewName.charAt(0).toUpperCase();
                                        
                                        // Update styling if moving from Unknown -> Known
                                        if (msg.classList.contains('Unknown') && !formattedNewName.startsWith('Unknown')) {
                                            msg.classList.remove('Unknown');
                                        }
                                    }
                                });
                                // 2. Refresh the roster
                                updateSpeakerRoster();
                            }
                        }
                    };
                    list.appendChild(li);
                });
            }

            function clearTranscript() {
                document.getElementById('transcript-box').innerHTML = '';
            }

            function downloadTranscript() {
                const msgs = document.querySelectorAll('.message-wrapper');
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