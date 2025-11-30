import asyncio
import threading
import time
import json
import os
from collections import defaultdict, deque
from flask import Flask, render_template_string, jsonify

from library.input_layer import TopicActivityMonitorMulti
from library.output_layer import OutputLayerMetadata, OutputLayerReceiver




class OutputLayerMonitor:
    """Runs a Kafka receiver in the background and exposes a Flask dashboard UI."""
    
    def __init__(self, source_name: str, service: str, broker: str = "152.53.32.66:9094"):
        self.source_name = source_name
        self.service = service
        self.broker = broker

        self.total_bytes = 0
        self.start_time = time.time()

        self.monitor = TopicActivityMonitorMulti("input.>")
        self.monitor_loop = asyncio.new_event_loop()
        threading.Thread(target=self._start_monitor_loop, daemon=True).start()

        # schedule monitor.connect() in that loop
        self.monitor_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self.monitor.connect())
        )

        self.service_buffers = defaultdict(lambda: deque(maxlen=500))

    
        self.storage_dir = "storage"
        os.makedirs(self.storage_dir, exist_ok=True)

        self.app = Flask(__name__)
        self._setup_routes()

    def _start_monitor_loop(self):
        asyncio.set_event_loop(self.monitor_loop)
        self.monitor_loop.run_forever()
    # --------------------------------------------------
    # Flask Routes
    # --------------------------------------------------
    def _setup_routes(self):

        # Main Dashboard
        @self.app.route("/")
        def index():
            return render_template_string(self._base_layout(self._dashboard_html()))

  
        @self.app.route("/api/stats")
        def get_stats():
            total = sum(len(buf) for buf in self.service_buffers.values())
            total_bytes = 0
            for buf in self.service_buffers.values():
                for entry in buf:
                    total_bytes += len(json.dumps(entry).encode("utf-8"))

            runtime_sec = time.time() - self.start_time

            return jsonify({
                "messages": total,
                "total_mb": round(total_bytes / (1024 * 1024), 3),
                "runtime_sec": round(runtime_sec, 1)
            })

    
        @self.app.route("/api/messages")
        def get_messages():
            result = []

            for service, buf in self.service_buffers.items():
                for entry in list(buf):
                    entry_copy = dict(entry)
                    entry_copy["byte_size"] = len(json.dumps(entry).encode("utf-8"))
                    result.append(entry_copy)

            return jsonify(result)

   
        @self.app.route("/services/input/monitor")
        def service_status_page():
            return render_template_string(self._base_layout(self._services_html()))

        @self.app.route("/api/services/input/monitor/<service_id>")
        def service_monitor(service_id):
            status = self.monitor.get_status()

            if service_id not in status:
                return jsonify({"error": "Service not found", "service_id": service_id}), 404

            return jsonify({service_id: status[service_id]})
        
        @self.app.route("/api/services/input/monitor")
        def service_monitor_all():
            return self.monitor.get_status()

        @self.app.route("/api/history/<service_id>")
        def get_service_history(service_id):
            buf = self.service_buffers.get(service_id)
            if not buf:
                return jsonify([])
            return jsonify(list(buf))

    
        @self.app.route("/rest-info")
        def rest_info():
            return render_template_string(self._base_layout(self._rest_info_html()))

    # --------------------------------------------------
    # Kafka Callback
    # --------------------------------------------------
    async def _msg_callback(self, metadata: OutputLayerMetadata):
        data = metadata.to_dict()


        service = data.get("service_id")

        # save in ringbuffer
        self.service_buffers[service].append(data)

        # persist to disk
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
    def _services_html(self):
        return """
    <style>
    body {
        font-family: Arial;
        margin: 20px;
        background: #fafafa;
    }

    h1 {
        margin-bottom: 20px;
    }

    table {
        width: 100%;
        background: white;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
    }

    th, td {
        padding: 14px;
        border-bottom: 1px solid #eee;
    }

    th {
        background: #f3f3f3;
        text-align: left;
    }

    .status-dot {
        height: 14px;
        width: 14px;
        border-radius: 50%;
        display: inline-block;
    }

    .status-online {
        background-color: #37d67a;
        box-shadow: 0 0 6px #37d67a;
    }

    .status-offline {
        background-color: #ff4d4f;
        box-shadow: 0 0 6px #ff4d4f;
    }

    #refreshTime {
        color: #777;
        font-size: 14px;
    }
    </style>

    <h1>Input Service Status</h1>
    <p id="refreshTime">Loading…</p>

    <table>
        <thead>
            <tr>
                <th>Status</th>
                <th>Input Service ID</th>
                <th>Last Seen</th>
                <th>Last Seen (Timestamp)</th>
            </tr>
        </thead>
        <tbody id="serviceTable">
            <tr><td colspan="4">Loading…</td></tr>
        </tbody>
    </table>

    <script>
    // Format Unix timestamp ->  readable
    function formatTime(ts) {
        if (!ts) return "-";
        const d = new Date(ts * 1000);
        return d.toLocaleString();
    }

    async function refresh() {
        const res = await fetch('/api/services/input/monitor');
        const status = await res.json();
        const tbody = document.getElementById('serviceTable');
        tbody.innerHTML = "";

        const now = Math.floor(Date.now() / 1000);
        document.getElementById("refreshTime").innerText =
            "Last refreshed: " + new Date().toLocaleTimeString();

        Object.entries(status).forEach(([service, info]) => {
            const isOnline = info.online;
            const lastSeen = info.last_seen ? formatTime(info.last_seen) : "-";
            const lastSeenRaw = info.last_seen ? info.last_seen : "-";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <span class="status-dot ${isOnline ? "status-online" : "status-offline"}"></span>
                </td>
                <td>${service}</td>
                <td>${lastSeen}</td>
                <td>${lastSeenRaw}</td>
            `;
            tbody.appendChild(tr);
        });

        if (Object.keys(status).length === 0) {
            tbody.innerHTML = `<tr><td colspan="4">No input services detected</td></tr>`;
        }
    }

    // Auto-refresh every 2 seconds
    setInterval(refresh, 2000);
    refresh();
    </script>

        """

    def _rest_info_html(self):
        return """
    <style>
    body {
        font-family: Arial, sans-serif;
        margin: 40px;
        max-width: 900px;
    }
    h1 {
        font-size: 32px;
        margin-bottom: 10px;
    }
    h2 {
        margin-top: 28px;
    }
    .endpoint {
        background: #f5f5f5;
        padding: 10px 14px;
        border-radius: 8px;
        font-family: Consolas, monospace;
        margin-top: 6px;
        display: inline-block;
    }
    .desc {
        color: #555;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    a {
        color: #007bff;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>

    <h1>REST API Übersicht</h1>
    <p>Hier findest du eine kurze Beschreibung aller verfügbaren REST-Endpunkte des Monitors.</p>


    <h2>Live Status aller Input-Services</h2>
    <div class="endpoint">GET /api/services/input/monitor</div>
    <div class="desc">Zeigt alle erkannten Input-Services mit <code>online</code>-Status und <code>last_seen</code>-Zeitstempel.</div>


    <h2>Status eines einzelnen Services</h2>
    <div class="endpoint">GET /api/services/input/monitor/&lt;service_id&gt;</div>
    <div class="desc">Details zu einem spezifischen Input-Service.</div>


    <h2>Historie (max. 500) eines Output-Services</h2>
    <div class="endpoint">GET /api/history/&lt;service_id&gt;</div>
    <div class="desc">Gibt die letzten 500 Nachrichten dieses Services aus dem Ringbuffer zurück.</div>


    <h2>Alle gespeicherten Output-Nachrichten (Ringbuffer)</h2>
    <div class="endpoint">GET /api/messages</div>
    <div class="desc">Liefert alle Nachrichten aus allen Service-Ringpuffern.</div>


    <h2>System-Statistiken</h2>
    <div class="endpoint">GET /api/stats</div>
    <div class="desc">Zeigt Anzahl der Nachrichten, Gesamtgröße und Laufzeit.</div>


    <p style="margin-top:40px;">
        <a href="/">← Zurück zum Dashboard</a>
    </p>
        """


    # --------------------------------------------------
    # Dashboard HTML (with burger menu)
    # --------------------------------------------------

    def _base_layout(self, body_content: str) -> str:
        return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Smart Interaction Monitor</title>

    <style>
    body {{
        font-family: Arial;
        margin: 0;
        background: #fafafa;
    }}

    #menuButton {{
        font-size: 30px;
        cursor: pointer;
        padding: 15px;
        position: fixed;
        z-index: 20;
        top: 0;
        left: 0;
        color: black; /* closed state */
    }}

    #menuButton.open {{
        color: white; /* open state */
    }}

    #sideMenu {{
        height: 100%;
        width: 0;
        position: fixed;
        z-index: 15;
        top: 0;
        left: 0;
        background-color: #222;
        overflow-x: hidden;
        transition: 0.3s;
        padding-top: 60px;
    }}

    #sideMenu a {{
        padding: 10px 20px;
        text-decoration: none;
        display: block;
        color: #ddd;
        font-size: 20px;
    }}

    #sideMenu a:hover {{
        background: #444;
    }}

    #sideMenuClose {{
        font-size: 30px;
        color: white;
        cursor: pointer;
        padding: 10px 20px;
    }}
    </style>

    </head>

    <body>

    <div id="menuButton" onclick="toggleMenu()">☰</div>

    <div id="sideMenu">
        <a href="/">Overview</a>
        <a href="/services/input/monitor">Service Monitor</a>
        <a href="/rest-info">REST API Info</a>
        <a href="/api/messages" target="_blank">/api/messages</a>
        <a href="/api/stats" target="_blank">/api/stats</a>
    </div>

    <script>
    document.addEventListener("DOMContentLoaded", function() {{

        let menuOpen = false;

        window.toggleMenu = function() {{
            menuOpen = !menuOpen;

            const sideMenu = document.getElementById("sideMenu");
            const btn = document.getElementById("menuButton");

            sideMenu.style.width = menuOpen ? "260px" : "0";

            // Farbe ändern
            if (menuOpen) {{
                btn.classList.add("open");
            }} else {{
                btn.classList.remove("open");
            }}
        }};

        document.querySelectorAll('#sideMenu a').forEach(a => {{
            a.addEventListener("click", () => {{
                menuOpen = false;
                document.getElementById("sideMenu").style.width = "0";
                document.getElementById("menuButton").classList.remove("open");
            }});
        }});

    }});
    </script>


    <div style="padding: 30px; margin-top: 50px;">
    {body_content}
    </div>

    </body>
    </html>
    """



    def _dashboard_html(self):
        return """
<style>
body {
    font-family: Arial;
    margin: 0;
    background: #fafafa;
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
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.tile-title { font-size: 14px; color: #777; }
.tile-value { font-size: 28px; font-weight: bold; }
.tile:hover {
    transform: translateY(-4px); /* leicht nach oben heben */
    box-shadow: 0 6px 15px rgba(0,0,0,0.15); /* stärkerer Schatten */
}
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
        """


monitor = OutputLayerMonitor(
    source_name="camera1",
    service="object_detection"
)

app = monitor.app   # Flask instance exposed for Gunicorn

def start_background():
    import threading
    import asyncio

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(monitor._receiver_loop())
        loop.close()

    threading.Thread(target=runner, daemon=True).start()

start_background()

# --- STANDALONE MODE ---
if __name__ == "__main__":
    # Start Flask dev server
    monitor.start(flask_port=5000)