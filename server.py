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
                --border: #334155;
                --danger: #ef4444;
                --success: #10b981;
                --unknown-color: #64748b;
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
            
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #64748b; }

            /* Sidebar */
            #sidebar { 
                width: 350px; 
                background: var(--panel); 
                border-right: 1px solid var(--border); 
                display: flex; flex-direction: column; 
                padding: 24px; gap: 22px; box-sizing: border-box; z-index: 10;
            }
            
            .header-container { display: flex; justify-content: space-between; align-items: center; }
            h1 { font-size: 20px; margin: 0; font-weight: 700; }
            
            .status-badge {
                display: flex; align-items: center; gap: 6px;
                padding: 5px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
                background: rgba(148, 163, 184, 0.1); color: var(--subtext);
            }
            .status-badge.live { background: rgba(16, 185, 129, 0.15); color: var(--success); }
            .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
            
            .section-title { font-size: 11px; text-transform: uppercase; color: var(--subtext); font-weight: 700; margin-bottom: 8px; }
            
            button, select, input { 
                width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); 
                background: rgba(15, 23, 42, 0.6); color: white; font-family: 'Inter', sans-serif; font-size: 13px;
                outline: none; transition: all 0.2s;
            }
            button { background: #273549; cursor: pointer; font-weight: 500; }
            button:hover { background: #334155; }
            button.primary { background: var(--primary); border-color: var(--primary); font-weight: 600; }
            button.danger { background: rgba(239, 68, 68, 0.15); color: var(--danger); border-color: rgba(239, 68, 68, 0.3); }
            button.danger:hover { background: var(--danger); color: white; }
            
            #speaker-list { list-style: none; padding: 0; margin: 0; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 6px; }
            #speaker-list li { 
                background: rgba(15, 23, 42, 0.4); padding: 8px 12px; border-radius: 6px; font-size: 13px; 
                display: flex; justify-content: space-between; align-items: center; cursor: pointer;
            }
            #speaker-list li:hover { background: rgba(15, 23, 42, 0.8); border-color: var(--primary); }
            .roster-color-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
            
            /* Main Content Area */
            #main { flex: 1; display: flex; flex-direction: column; background: var(--bg); position: relative; }
            #header { 
                display: flex; justify-content: space-between; align-items: center; 
                padding: 20px 36px; border-bottom: 1px solid var(--border); background: rgba(11, 17, 32, 0.8);
                backdrop-filter: blur(8px);
            }
            #header h2 { margin: 0; font-size: 17px; font-weight: 600; }
            .header-buttons { display: flex; gap: 10px; }
            .header-buttons button { width: auto; padding: 6px 14px; font-size: 12px; border-radius: 18px; }
            
            #transcript-box { 
                flex: 1; overflow-y: auto; display: flex; flex-direction: column; 
                gap: 16px; padding: 24px 36px; scroll-behavior: smooth;
            }
            
            /* Message Styles */
            .message-wrapper { display: flex; gap: 14px; animation: fadeIn 0.25s ease forwards; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
            
            .avatar {
                width: 36px; height: 36px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center; font-weight: 600; 
                font-size: 14px; color: white; flex-shrink: 0;
            }
            
            .message-content {
                display: flex; flex-direction: column; gap: 4px; max-width: 85%;
                background: var(--panel); padding: 14px 18px; border-radius: 0 14px 14px 14px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            
            .msg-header { display: flex; align-items: baseline; gap: 10px; }
            .speaker-tag { font-weight: 600; font-size: 13px; }
            .msg-meta { color: #64748b; font-size: 11px; }
            .msg-body { font-size: 14.5px; line-height: 1.55; color: #e2e8f0; }

            /* Speculative Placeholder */
            .placeholder-wrapper { display: flex; gap: 14px; opacity: 0.7; animation: pulse 1.5s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.8; } }
            .placeholder-body { display: flex; align-items: center; gap: 6px; font-style: italic; color: #94a3b8; font-size: 13px; }
            .spinner-dot { width: 5px; height: 5px; background: #94a3b8; border-radius: 50%; display: inline-block; }
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
                <div class="section-title">Settings</div>
                <select id="audio-source" onchange="changeAudioSource()" style="margin-bottom: 10px;">
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
                <div class="section-title">Speaker Enrollment</div>
                <div style="display: flex; gap: 6px;">
                    <input type="text" id="speaker-name" placeholder="Enter name...">
                    <button onclick="enrollSpeaker()" style="flex: 0 0 75px;">Enroll</button>
                </div>
                <div id="enroll-status" style="font-size: 11px; color: var(--success); margin-top: 5px; height: 14px;"></div>
            </div>

            <div style="display: flex; flex-direction: column; flex: 1; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div class="section-title" style="margin: 0;">Roster (Click to Rename)</div>
                    <button onclick="clearSpeakers()" style="width: auto; padding: 2px 8px; font-size: 10px; border-radius: 10px;">Clear All</button>
                </div>
                <ul id="speaker-list"></ul>
            </div>
        </div>

        <div id="main">
            <div id="header">
                <h2>Live Meeting Transcript</h2>
                <div class="header-buttons">
                    <button onclick="downloadTranscript()" style="background: rgba(255,255,255,0.08); border:none;">Download .TXT</button>
                    <button onclick="clearTranscript()" style="background: rgba(255,255,255,0.08); border:none;">Clear View</button>
                </div>
            </div>
            <div id="transcript-box"></div>
        </div>

        <script>
            let isRunning = false;
            let expectedSeq = 0;
            const pendingChunks = {}; 
            let ws;

            let lastSpeaker = null;
            let lastEpoch = 0;
            let currentBubbleWordCount = 0;
            let activeBubbleElement = null;

            // --- DYNAMIC SPEAKER COLOR PALETTE ---
            const colorPalette = [
                '#3b82f6', // Blue
                '#10b981', // Emerald
                '#8b5cf6', // Violet
                '#f59e0b', // Amber
                '#ec4899', // Pink
                '#14b8a6', // Teal
                '#ef4444', // Red
                '#f97316'  // Orange
            ];
            const speakerColors = {};
            let colorIndex = 0;

            function getSpeakerColor(speakerName) {
                if (speakerName.startsWith('Unknown')) {
                    return '#64748b'; // Slate gray for unknowns
                }
                if (!speakerColors[speakerName]) {
                    speakerColors[speakerName] = colorPalette[colorIndex % colorPalette.length];
                    colorIndex++;
                }
                return speakerColors[speakerName];
            }

            function initWebSocket() {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws`);

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);

                    if (data.type === "pending") {
                        showPlaceholder(data);
                    } else if (data.type === "cancel") {
                        removePlaceholder(data.seq);
                    } else if (data.type === "transcript") {
                        removePlaceholder(data.seq);
                        
                        pendingChunks[data.seq] = data;
                        while (pendingChunks[expectedSeq]) {
                            renderTranscriptChunk(pendingChunks[expectedSeq]);
                            delete pendingChunks[expectedSeq];
                            expectedSeq++;
                        }
                    }
                };
                ws.onclose = () => setTimeout(initWebSocket, 2000);
            }

            function showPlaceholder(data) {
                const box = document.getElementById('transcript-box');
                if (document.getElementById(`placeholder-${data.seq}`)) return;

                const pDiv = document.createElement('div');
                pDiv.id = `placeholder-${data.seq}`;
                pDiv.className = 'placeholder-wrapper';
                pDiv.innerHTML = `
                    <div class="avatar" style="background: #334155;">...</div>
                    <div class="message-content" style="background: rgba(30, 41, 59, 0.4);">
                        <div class="placeholder-body">
                            <span>Transcribing audio</span>
                            <span class="spinner-dot"></span>
                        </div>
                    </div>
                `;
                box.appendChild(pDiv);
                box.scrollTop = box.scrollHeight;
            }

            function removePlaceholder(seq) {
                const el = document.getElementById(`placeholder-${seq}`);
                if (el) el.remove();
            }

            function renderTranscriptChunk(item) {
                const box = document.getElementById('transcript-box');
                const words = item.text.trim().split(/\s+/).length;

                const isSameSpeaker = (item.speaker === lastSpeaker);
                // Tightened exchange limit: club if spoken within 10 seconds of previous chunk
                const isQuickExchange = ((item.epoch - lastEpoch) <= 10.0); 
                const isUnderWordLimit = ((currentBubbleWordCount + words) <= 150); 

                const speakerColor = getSpeakerColor(item.speaker);

                if (isSameSpeaker && isQuickExchange && isUnderWordLimit && activeBubbleElement) {
                    const bodyEl = activeBubbleElement.querySelector('.msg-body');
                    bodyEl.textContent += " " + item.text;
                    
                    const currentTotalText = activeBubbleElement.getAttribute('data-text');
                    activeBubbleElement.setAttribute('data-text', currentTotalText + " " + item.text);
                    currentBubbleWordCount += words;
                } else {
                    const div = document.createElement('div');
                    const isUnknown = item.speaker.startsWith('Unknown');
                    
                    div.className = `message-wrapper ${isUnknown ? 'Unknown' : ''}`;
                    div.setAttribute('data-speaker', item.speaker);
                    div.setAttribute('data-time', item.timestamp);
                    div.setAttribute('data-text', item.text);

                    const initial = isUnknown ? '?' : item.speaker.charAt(0).toUpperCase();

                    div.innerHTML = `
                        <div class="avatar" style="background-color: ${speakerColor};">${initial}</div>
                        <div class="message-content">
                            <div class="msg-header">
                                <span class="speaker-tag" style="color: ${speakerColor};">${item.speaker}</span>
                                <span class="msg-meta">${item.timestamp}</span>
                            </div>
                            <div class="msg-body">${item.text}</div>
                        </div>
                    `;
                    
                    box.appendChild(div);
                    activeBubbleElement = div;
                    currentBubbleWordCount = words;
                    lastSpeaker = item.speaker;
                }

                lastEpoch = item.epoch;
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
                    lastSpeaker = null;
                    activeBubbleElement = null;
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
                    status.textContent = `Enrolled!`;
                    input.value = '';
                    updateSpeakerRoster();
                    setTimeout(() => status.textContent = '', 3000);
                }
            }

            async function clearSpeakers() {
                if(confirm("Clear all enrolled speaker profiles?")) {
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
                    const color = getSpeakerColor(spk); // Fetch dynamic color
                    
                    li.innerHTML = `
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="roster-color-dot" style="background-color: ${color};"></span>
                            <span>${spk}</span>
                        </div>
                        <span style="font-size: 11px; opacity: 0.6;">✎ Edit</span>
                    `;
                    
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
                                // Retrieve the new color mapped to the updated name
                                const newColor = getSpeakerColor(formattedNewName);
                                
                                document.querySelectorAll('.message-wrapper').forEach(msg => {
                                    if (msg.getAttribute('data-speaker') === spk) {
                                        msg.setAttribute('data-speaker', formattedNewName);
                                        
                                        const tag = msg.querySelector('.speaker-tag');
                                        tag.textContent = formattedNewName;
                                        tag.style.color = newColor;
                                        
                                        const avatar = msg.querySelector('.avatar');
                                        avatar.textContent = formattedNewName.charAt(0).toUpperCase();
                                        avatar.style.backgroundColor = newColor;
                                        
                                        if (msg.classList.contains('Unknown') && !formattedNewName.startsWith('Unknown')) {
                                            msg.classList.remove('Unknown');
                                        }
                                    }
                                });
                                updateSpeakerRoster();
                            }
                        }
                    };
                    list.appendChild(li);
                });
            }

            function clearTranscript() {
                document.getElementById('transcript-box').innerHTML = '';
                lastSpeaker = null;
                activeBubbleElement = null;
            }

            function downloadTranscript() {
                const msgs = document.querySelectorAll('.message-wrapper');
                let textContent = "WispNotes Meeting Transcript\\n==============================\\n\\n";
                
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
                a.download = `Transcript_${new Date().toISOString().slice(0,10)}.txt`;
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