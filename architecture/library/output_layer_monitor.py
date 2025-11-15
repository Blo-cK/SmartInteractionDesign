import asyncio
import threading
import time
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify

from output_layer import OutputLayerReceiver, OutputLayerMetadata


class OutputLayerMonitor:
    """Runs a Kafka receiver in the background and exposes a Flask dashboard UI."""
    
    def __init__(self, source_name: str, service: str, broker: str = "152.53.32.66:9094"):
        self.source_name = source_name
        self.service = service
        self.broker = broker

        self.received_messages = []     # List[OutputLayerMetadata]
        self.total_bytes = 0
        self.start_time = time.time()

        self.app = Flask(__name__)

        # Flask routes
        self._setup_routes()

    # --------------------------------------------------
    # Flask Website
    # --------------------------------------------------
    def _setup_routes(self):

        @self.app.route("/")
        def index():
            return render_template_string(self._dashboard_html())

        @self.app.route("/api/stats")
        def get_stats():
            runtime_sec = time.time() - self.start_time
            return jsonify({
                "messages": len(self.received_messages),
                "total_mb": round(self.total_bytes / (1024 * 1024), 3),
                "runtime_sec": round(runtime_sec, 1)
            })

        @self.app.route("/api/messages")
        def get_messages():
            msgs = [m.to_dict() for m in self.received_messages]
            # add byte size per message
            for m in msgs:
                m["byte_size"] = len(json.dumps(m).encode("utf-8"))
            return jsonify(msgs)

    # --------------------------------------------------
    # Kafka Receiver Background Task
    # --------------------------------------------------
    async def _msg_callback(self, metadata: OutputLayerMetadata):
        data = metadata.to_dict()
        data_bytes = len(json.dumps(data).encode("utf-8"))

        self.total_bytes += data_bytes
        self.received_messages.append(metadata)

    async def _receiver_loop(self):
        receiver = OutputLayerReceiver(broker=self.broker, group_id=None)

        try:
            
            await receiver.receiveAllMetadata(self._msg_callback)
        finally:
            await receiver.disconnect()

    # --------------------------------------------------
    # Start Monitor
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
    # Dashboard HTML
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
        margin: 30px; 
        background: #fafafa;
    }

    h1 { 
        margin-top: 0; 
        font-size: 32px;
    }

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
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .tile:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }

    .tile-title {
        font-size: 14px;
        color: #777;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .tile-value {
        font-size: 28px;
        font-weight: bold;
        margin-top: 5px;
    }

    table { 
        width: 100%; 
        border-collapse: collapse; 
        margin-top: 20px;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
    }

    th, td { 
        padding: 12px; 
        border-bottom: 1px solid #eee; 
    }

    th {
        background: #f7f7f7;
        font-weight: bold;
        font-size: 14px;
        text-align: left;
    }

    tr:hover { 
        background: #f9f9f9; 
    }

    button { 
        padding: 6px 10px; 
        border: none;
        border-radius: 6px;
        cursor: pointer;
        background: #007bff;
        color: white;
        transition: background 0.15s ease;
    }

    button:hover {
        background: #0056d6;
    }
</style>

</head>
<body>

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
<div id="popup" 
     style="display:none; position:fixed; left:0; top:0; width:100%; height:100%; 
            background:rgba(0,0,0,0.5); justify-content:center; align-items:center;">
    <div style="background:white; padding:20px; border-radius:8px; width:400px;">
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

async function refreshStats() {
    const res = await fetch("/api/stats");
    const s = await res.json();
    document.getElementById("msg_count").innerText = s.messages;
    document.getElementById("msg_mb").innerText = s.total_mb;
    document.getElementById("runtime").innerText = s.runtime_sec;
}

async function refreshTable() {
    const res = await fetch("/api/messages");
    const msgs = await res.json();

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

</body>
</html>
        """


# ------------------------------------------------------
# Run standalone
# ------------------------------------------------------
if __name__ == "__main__":
    print("Starting monitor...")
    monitor = OutputLayerMonitor(
        source_name="camera1",
        service="object_detection"
    )
    monitor.start(flask_port=5000)
