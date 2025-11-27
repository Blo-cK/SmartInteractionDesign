import asyncio
import threading
import time
import json
import os
from collections import defaultdict, deque
from flask import Flask, render_template_string, jsonify

from architecture.library.output_layer import OutputLayerReceiver, OutputLayerMetadata


class OutputLayerMonitor:
    """Runs a Kafka receiver in the background and exposes a Flask dashboard UI."""
    
    def __init__(self, source_name: str, service: str, broker: str = "152.53.32.66:9094"):
        self.source_name = source_name
        self.service = service
        self.broker = broker

        self.received_messages = []     
        self.total_bytes = 0
        self.start_time = time.time()

        # NEW: Per-service ring buffer (500 entries max)
        self.service_buffers = defaultdict(lambda: deque(maxlen=500))

        # NEW: JSONL persistent storage
        self.storage_dir = "storage"
        os.makedirs(self.storage_dir, exist_ok=True)

        self.app = Flask(__name__)
        self._setup_routes()

    # --------------------------------------------------
    # Flask Routes
    # --------------------------------------------------
    def _setup_routes(self):

        # Main Dashboard
        @self.app.route("/")
        def index():
            return render_template_string(self._dashboard_html())

        # REST: Stats
        @self.app.route("/api/stats")
        def get_stats():
            runtime_sec = time.time() - self.start_time
            return jsonify({
                "messages": len(self.received_messages),
                "total_mb": round(self.total_bytes / (1024 * 1024), 3),
                "runtime_sec": round(runtime_sec, 1)
            })

        # REST: All messages (live)
        @self.app.route("/api/messages")
        def get_messages():
            msgs = [m.to_dict() for m in self.received_messages]
            for m in msgs:
                m["byte_size"] = len(json.dumps(m).encode("utf-8"))
            return jsonify(msgs)

        # REST: List all services
        @self.app.route("/api/services")
        def list_services():
            return jsonify(sorted(list(self.service_buffers.keys())))

        # REST: Return last 500 messages for a service
        @self.app.route("/api/history/<service_id>")
        def get_service_history(service_id):
            buf = self.service_buffers.get(service_id)
            if not buf:
                return jsonify([])
            return jsonify(list(buf))

        # NEW: Static page that documents the REST interface
        @self.app.route("/rest-info")
        def rest_info():
            return render_template_string(self._rest_info_html())

    # --------------------------------------------------
    # Kafka Callback
    # --------------------------------------------------
    async def _msg_callback(self, metadata: OutputLayerMetadata):
        data = metadata.to_dict()
        data_bytes = len(json.dumps(data).encode("utf-8"))

        # existing behaviour
        self.total_bytes += data_bytes
        self.received_messages.append(metadata)

        service = data.get("service_id")

        # NEW: write into memory buffer
        self.service_buffers[service].append(data)

        # NEW: write into JSONL
        file_path = os.path.join(self.storage_dir, f"{service}.jsonl")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    async def _receiver_loop(self):
        receiver = OutputLayerReceiver(broker=self.broker, group_id=None)
        try:
            await receiver.receiveAllData(self._msg_callback)
        finally:
            await receiver.disconnect()

    # --------------------------------------------------
    # Start
    # --------------------------------------------------
    def start(self, flask_port: int = 5000):

        threading.Thread(
            target=lambda: asyncio.run(self._receiver_loop()),
            daemon=True
        ).start()

        print("[Monitor] Kafka consumer running...")
        print(f"[Monitor] Flask running at http://localhost:{flask_port}")

        self.app.run(host="0.0.0.0", port=flask_port)

    # --------------------------------------------------
    # REST API Documentation Page
    # --------------------------------------------------
    def _rest_info_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
<title>REST API Dokumentation</title>

<style>
body {
    font-family: Arial;
    margin: 40px;
}
h1 { font-size: 32px; }
code {
    background: #eee;
    padding: 4px 6px;
    border-radius: 6px;
}
a { color: #007bff; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>

</head>
<body>

<h1>REST API Übersicht</h1>

<p>Hier findest du die verfügbaren REST-Endpunkte des OutputLayer-Monitors.</p>

<h2> Liste aller Services</h2>
<code>GET /api/services</code>

<h2> Letzte 500 Einträge eines Services</h2>
<code>GET /api/history/&lt;service_id&gt;</code>

<h3>Beispiel:</h3>
<code>GET /api/history/chatbot</code>

<h2>Übersicht aller Live Messages</h2>
<code>GET /api/messages</code>

<h2> System-Statistiken</h2>
<code>GET /api/stats</code>

<p><br><a href="/"> Zurück zum Dashboard</a></p>

</body>
</html>
        """

    # --------------------------------------------------
    # Dashboard HTML (with burger menu)
    # --------------------------------------------------
    def _dashboard_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
<title>OutputLayer Monitor</title>

<style>
body {
    font-family: Arial;
    margin: 0;
    background: #fafafa;
}

/* ----------------------- */
/* BURGER MENU */
/* ----------------------- */

#menuButton {
    font-size: 30px;
    cursor: pointer;
    padding: 15px;
    position: absolute;
}

#sideMenu {
    height: 100%;
    width: 0;
    position: fixed;
    z-index: 10;
    top: 0;
    left: 0;
    background-color: #222;
    overflow-x: hidden;
    transition: 0.3s;
    padding-top: 60px;
}

#sideMenu a {
    padding: 10px 20px;
    text-decoration: none;
    display: block;
    color: #ddd;
    font-size: 20px;
}

#sideMenu a:hover {
    background: #444;
}

#sideMenuClose {
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 36px;
    color: white;
    cursor: pointer;
}

/* ----------------------- */

.content {
    padding: 30px;
}

/* tile style */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 25px;
}
.tile {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
}
.tile-title { font-size: 14px; color: #777; }
.tile-value { font-size: 28px; font-weight: bold; }

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    background: white;
    border-radius: 10px;
    overflow: hidden;
}
th, td { padding: 12px; border-bottom: 1px solid #eee; }
th { background: #f7f7f7; }

/* Popup */
#popup {
    display:none; position:fixed; left:0; top:0; width:100%; height:100%;
    background:rgba(0,0,0,0.5); justify-content:center; align-items:center;
}
#popupBox {
    background:white; padding:20px; border-radius:8px; width:400px;
}
</style>
</head>

<body>

<div id="menuButton" onclick="openMenu()">☰</div>

<!-- SIDE MENU -->
<div id="sideMenu">
    <div id="sideMenuClose" onclick="closeMenu()"></div>
    <a href="/">Overview</a>
    <a href="/rest-info">REST API Info</a>
    <a href="/api/services" target="_blank"> /api/services</a>
    <a href="/api/messages" target="_blank">/api/messages</a>
    <a href="/api/stats" target="_blank"> /api/stats</a>
</div>

<script>
function openMenu() {
    document.getElementById("sideMenu").style.width = "260px";
}
function closeMenu() {
    document.getElementById("sideMenu").style.width = "0";
}
</script>

<div class="content">
<h1>OutputLayer Monitoring</h1>

<div class="stats-grid">
    <div class="tile">
        <div class="tile-title">Nachrichten</div>
        <div class="tile-value" id="msg_count">0</div>
    </div>
    <div class="tile">
        <div class="tile-title">Gesamtgröße</div>
        <div class="tile-value"><span id="msg_mb">0</span> MB</div>
    </div>
    <div class="tile">
        <div class="tile-title">Laufzeit</div>
        <div class="tile-value"><span id="runtime">0</span> sec</div>
    </div>
</div>

<h2>Empfangene Metadaten</h2>

<label for="serviceFilter"><b>Service filtern:</b></label>
<select id="serviceFilter" onchange="refreshTable()">
    <option value="">Alle</option>
</select>

<table id="msg_table">
    <thead>
        <tr>
            <th>Source</th>
            <th>Service</th>
            <th>Timestamp (input)</th>
            <th>Completed at</th>
            <th>Size</th>
            <th>Result</th>
        </tr>
    </thead>
    <tbody></tbody>
</table>

<!-- Popup -->
<div id="popup">
    <div id="popupBox">
        <pre id="popup_content"></pre>
        <button onclick="closePopup()">Schließen</button>
    </div>
</div>

<script>
function openPopup(content) {
    document.getElementById("popup_content").innerText = content;
    document.getElementById("popup").style.display = "flex";
}
function closePopup() {
    document.getElementById("popup").style.display = "none";
}

function updateServiceDropdown(msgs) {
    const dropdown = document.getElementById("serviceFilter");
    const current = dropdown.value;
    const services = [...new Set(msgs.map(m => m.service_id))];

    dropdown.innerHTML = `<option value="">Alle</option>`;
    services.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.innerText = s;
        if (s === current) opt.selected = true;
        dropdown.appendChild(opt);
    });
}

async function refreshStats() {
    const res = await fetch("/api/stats");
    const s = await res.json();
    document.getElementById("msg_count").innerText = s.messages;
    document.getElementById("msg_mb").innerText = s.total_mb;
    document.getElementById("runtime").innerText = s.runtime_sec;
}

async function refreshTable() {
    const res = await fetch("/api/messages");
    let msgs = await res.json();

    updateServiceDropdown(msgs);

    const selected = document.getElementById("serviceFilter").value;
    const exists = msgs.some(m => m.service_id === selected);

    if (selected && !exists) {
        document.getElementById("serviceFilter").value = "";
    } else if (selected) {
        msgs = msgs.filter(m => m.service_id === selected);
    }

    const body = document.querySelector("#msg_table tbody");
    body.innerHTML = "";

    msgs.forEach(m => {
        const jsonText = encodeURIComponent(JSON.stringify(m.result, null, 2));
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${m.source_id}</td>
            <td>${m.service_id}</td>
            <td>${m.time_stamp}</td>
            <td>${m.completed_at}</td>
            <td>${m.byte_size} B</td>
            <td><button onclick="openPopup(decodeURIComponent('${jsonText}'))">View</button></td>
        `;
        body.appendChild(tr);
    });
}

setInterval(() => {
    refreshStats();
    refreshTable();
}, 1500);
</script>

</div>

</body>
</html>
        """


