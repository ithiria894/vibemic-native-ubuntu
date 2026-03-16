#!/usr/bin/env python3
"""VibeMic Native — Voice-to-text for Ubuntu. Press PgDn to record, PgDn again to transcribe and type."""

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from openai import OpenAI
from pynput import keyboard
from PIL import Image, ImageDraw

# ─── Paths ───
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
ENV_FILE = SCRIPT_DIR / ".env"
TEMP_DIR = Path.home() / ".cache" / "vibemic"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_WAV = TEMP_DIR / "recording.wav"
HISTORY_FILE = SCRIPT_DIR / "history.json"
MIN_FILE_SIZE = 1000  # bytes — smaller means no real audio

# ─── Available models & options ───
WHISPER_MODELS = [
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
]

LANGUAGES = [
    ("Auto-detect", ""),
    ("English", "en"),
    ("廣東話 / Chinese", "zh"),
    ("日本語", "ja"),
    ("한국어", "ko"),
    ("Français", "fr"),
    ("Deutsch", "de"),
    ("Español", "es"),
    ("Português", "pt"),
    ("Italiano", "it"),
    ("Nederlands", "nl"),
    ("Polski", "pl"),
    ("Русский", "ru"),
    ("Türkçe", "tr"),
    ("العربية", "ar"),
    ("हिन्दी", "hi"),
    ("ภาษาไทย", "th"),
    ("Tiếng Việt", "vi"),
]

RESPONSE_FORMATS = ["json", "text", "srt", "verbose_json", "vtt"]

# ─── Config management ───
DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-4o-transcribe",
    "language": "",
    "prompt": "廣東話、English、普通話、日本語",
    "temperature": 0,
    "response_format": "json",
    "hotkey": "page_down",
}


def load_config():
    """Load config from config.json, falling back to .env for API key."""
    config = dict(DEFAULT_CONFIG)

    # Try config.json first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    # If no API key in config.json, try .env
    if not config.get("api_key"):
        config["api_key"] = _load_env_api_key()

    # Also check environment variable
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        config["api_key"] = env_key

    return config


def _load_env_api_key():
    """Read OPENAI_API_KEY from .env file."""
    if not ENV_FILE.exists():
        return ""
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def save_config(config):
    """Save config to config.json."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Failed to save config: {e}")


# ─── History ───
def load_history():
    """Load transcript history from history.json."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_to_history(text):
    """Append a transcript to history."""
    from datetime import datetime
    history = load_history()
    history.insert(0, {
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    # Keep last 200 entries
    history = history[:200]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def delete_history_entry(index):
    """Delete a single history entry by index."""
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except OSError:
            pass


def clear_history():
    """Clear all history."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
    except OSError:
        pass


# ─── State ───
config = load_config()
recording_process = None
is_recording = False
state_lock = threading.Lock()
RECORD_KEY = getattr(keyboard.Key, config.get("hotkey", "page_down"), keyboard.Key.page_down)


def notify(title, message, icon="dialog-information"):
    """Log to console only — no desktop notifications."""
    print(f"[{title}] {message}")


def create_tray_icon(color):
    """Create a tray icon: colored circle with white mic silhouette."""
    S = 48
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mc = (255, 255, 255, 240)

    # Background circle
    d.ellipse([0, 0, S - 1, S - 1], fill=color)

    # Mic body
    d.rounded_rectangle([17, 7, 31, 27], radius=7, fill=mc)

    # Arc (cup under mic)
    d.arc([11, 16, 37, 38], 0, 180, fill=mc, width=3)

    # Stand
    d.line([24, 37, 24, 42], fill=mc, width=3)

    # Base
    d.line([18, 42, 30, 42], fill=mc, width=3)

    return img


# ─── Settings GUI (browser-based, zero extra deps) ───
import http.server
import urllib.parse
import webbrowser
import socket

SETTINGS_PORT = None  # assigned dynamically


def _build_settings_html(cfg):
    """Generate the settings HTML page."""
    model_options = "".join(
        f'<option value="{m}"{"selected" if m == cfg.get("model", "") else ""}>{m}</option>'
        for m in WHISPER_MODELS
    )
    lang_options = "".join(
        f'<option value="{code}"{"selected" if code == cfg.get("language", "") else ""}>{name}</option>'
        for name, code in LANGUAGES
    )
    fmt_options = "".join(
        f'<option value="{f}"{"selected" if f == cfg.get("response_format", "") else ""}>{f}</option>'
        for f in RESPONSE_FORMATS
    )
    api_key_escaped = cfg.get("api_key", "").replace("&", "&amp;").replace('"', "&quot;")
    prompt_escaped = cfg.get("prompt", "").replace("&", "&amp;").replace("<", "&lt;")
    temp_val = cfg.get("temperature", 0)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>VibeMic Settings</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 30px; }}
  .container {{ max-width: 520px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 24px; color: #64b5f6; }}
  label {{ display: block; font-weight: 600; margin-bottom: 4px; margin-top: 16px; font-size: 14px; }}
  input[type=text], input[type=password], select, textarea {{
    width: 100%; padding: 10px 12px; border: 1px solid #333; border-radius: 6px;
    background: #16213e; color: #e0e0e0; font-size: 14px; outline: none;
  }}
  input:focus, select:focus, textarea:focus {{ border-color: #64b5f6; }}
  textarea {{ resize: vertical; min-height: 60px; font-family: inherit; }}
  .key-row {{ display: flex; gap: 8px; align-items: center; }}
  .key-row input {{ flex: 1; }}
  .key-row button {{ padding: 10px 14px; border: 1px solid #333; border-radius: 6px; background: #16213e; color: #aaa; cursor: pointer; font-size: 13px; }}
  .key-row button:hover {{ background: #1a1a3e; }}
  .slider-row {{ display: flex; align-items: center; gap: 12px; }}
  .slider-row input[type=range] {{ flex: 1; accent-color: #64b5f6; }}
  .slider-val {{ min-width: 32px; text-align: center; font-weight: 600; }}
  .btn-row {{ margin-top: 28px; text-align: right; }}
  .btn {{ padding: 10px 24px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 600; }}
  .btn-save {{ background: #64b5f6; color: #1a1a2e; }}
  .btn-save:hover {{ background: #90caf9; }}
  .msg {{ margin-top: 16px; padding: 12px; border-radius: 6px; display: none; font-size: 14px; }}
  .msg.ok {{ display: block; background: #1b5e20; color: #a5d6a7; }}
  .msg.err {{ display: block; background: #b71c1c; color: #ef9a9a; }}
  .hint {{ font-size: 12px; color: #888; margin-top: 2px; }}
</style>
</head><body>
<div class="container">
  <h1>VibeMic Settings</h1>
  <form id="f">
    <label>OpenAI API Key</label>
    <div class="key-row">
      <input type="password" id="api_key" name="api_key" value="{api_key_escaped}" placeholder="sk-...">
      <button type="button" onclick="let e=document.getElementById('api_key');e.type=e.type==='password'?'text':'password';this.textContent=e.type==='password'?'Show':'Hide'">Show</button>
    </div>

    <label>Model</label>
    <select name="model">{model_options}</select>

    <label>Language</label>
    <select name="language">{lang_options}</select>

    <label>Prompt <span class="hint">(hint for Whisper — e.g. expected languages)</span></label>
    <textarea name="prompt" rows="2">{prompt_escaped}</textarea>

    <label>Temperature</label>
    <div class="slider-row">
      <input type="range" name="temperature" min="0" max="1" step="0.1" value="{temp_val}"
             oninput="document.getElementById('tv').textContent=this.value">
      <span class="slider-val" id="tv">{temp_val}</span>
    </div>

    <label>Response Format</label>
    <select name="response_format">{fmt_options}</select>

    <div class="btn-row">
      <button type="submit" class="btn btn-save">Save</button>
    </div>
  </form>
  <div id="msg" class="msg"></div>
</div>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const fd = new FormData(e.target);
  const data = Object.fromEntries(fd.entries());
  data.temperature = parseFloat(data.temperature);
  const msg = document.getElementById('msg');
  try {{
    const r = await fetch('/save', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(data)}});
    if (r.ok) {{
      msg.className = 'msg ok'; msg.textContent = 'Settings saved!';
    }} else {{
      msg.className = 'msg err'; msg.textContent = 'Save failed: ' + (await r.text());
    }}
  }} catch(ex) {{
    msg.className = 'msg err'; msg.textContent = 'Connection error.';
  }}
}});
</script>
</body></html>"""


def _build_history_html():
    """Generate the history HTML page."""
    history = load_history()
    if not history:
        entries_html = '<p class="empty">No transcripts yet. Press PgDn to record!</p>'
    else:
        rows = []
        for i, entry in enumerate(history):
            text = entry.get("text", "")
            ts = entry.get("timestamp", "")
            text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
            text_json = json.dumps(text)
            rows.append(f"""
            <div class="entry" id="entry-{i}">
              <div class="entry-header">
                <span class="ts">{ts}</span>
                <span class="actions">
                  <button class="btn-copy" onclick='doCopy({text_json}, this)'>Copy</button>
                  <button class="btn-del" onclick="doDelete({i})">Delete</button>
                </span>
              </div>
              <div class="entry-text">{text_escaped}</div>
            </div>""")
        entries_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>VibeMic History</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 30px; }}
  .container {{ max-width: 680px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
  h1 {{ font-size: 22px; color: #64b5f6; }}
  .header-actions {{ display: flex; gap: 8px; }}
  .btn-settings, .btn-clear {{ padding: 8px 16px; border: 1px solid #333; border-radius: 6px;
    background: #16213e; color: #aaa; cursor: pointer; font-size: 13px; text-decoration: none; }}
  .btn-settings:hover {{ background: #1a1a3e; color: #e0e0e0; }}
  .btn-clear {{ color: #ef9a9a; border-color: #5a2020; }}
  .btn-clear:hover {{ background: #3a1010; }}
  .count {{ font-size: 13px; color: #888; margin-bottom: 16px; }}
  .entry {{ background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px; padding: 14px; margin-bottom: 10px; }}
  .entry-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .ts {{ font-size: 12px; color: #888; }}
  .actions {{ display: flex; gap: 6px; }}
  .btn-copy, .btn-del {{ padding: 4px 12px; border: 1px solid #333; border-radius: 4px;
    background: #1a1a2e; color: #aaa; cursor: pointer; font-size: 12px; }}
  .btn-copy:hover {{ background: #1b3a5e; color: #64b5f6; border-color: #64b5f6; }}
  .btn-del:hover {{ background: #3a1010; color: #ef9a9a; border-color: #ef9a9a; }}
  .btn-copied {{ background: #1b5e20 !important; color: #a5d6a7 !important; border-color: #2e7d32 !important; }}
  .entry-text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
  .empty {{ color: #888; font-size: 15px; text-align: center; padding: 40px; }}
</style>
</head><body>
<div class="container">
  <div class="header">
    <h1>VibeMic History</h1>
    <div class="header-actions">
      <a href="/" class="btn-settings">Settings</a>
      <button class="btn-clear" onclick="doClear()">Clear All</button>
    </div>
  </div>
  <div class="count">{len(history)} transcript{"s" if len(history) != 1 else ""}</div>
  <div id="entries">{entries_html}</div>
</div>
<script>
async function doCopy(text, btn) {{
  try {{
    await navigator.clipboard.writeText(text);
  }} catch(e) {{
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy');
    document.body.removeChild(ta);
  }}
  btn.textContent = 'Copied!';
  btn.classList.add('btn-copied');
  setTimeout(() => {{ btn.textContent = 'Copy'; btn.classList.remove('btn-copied'); }}, 1500);
}}

async function doDelete(index) {{
  const r = await fetch('/history/delete', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{index: index}})
  }});
  if (r.ok) {{
    document.getElementById('entry-' + index).style.display = 'none';
  }}
}}

async function doClear() {{
  if (!confirm('Clear all transcript history?')) return;
  const r = await fetch('/history/clear', {{method: 'POST'}});
  if (r.ok) location.reload();
}}
</script>
</body></html>"""


class _SettingsHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for settings + history pages."""

    on_save_callback = None

    def log_message(self, *args):
        pass  # suppress logs

    def do_GET(self):
        if self.path.startswith("/history"):
            html = _build_history_html()
        else:
            cfg = load_config()
            html = _build_settings_html(cfg)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if self.path == "/save":
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                self.send_error(400, "Invalid JSON")
                return

            api_key = data.get("api_key", "").strip()
            if not api_key:
                self.send_error(400, "API Key is required")
                return

            new_config = {
                "api_key": api_key,
                "model": data.get("model", "whisper-1"),
                "language": data.get("language", ""),
                "prompt": data.get("prompt", "").strip(),
                "temperature": round(float(data.get("temperature", 0)), 1),
                "response_format": data.get("response_format", "json"),
                "hotkey": "page_down",
            }
            save_config(new_config)
            if self.on_save_callback:
                self.on_save_callback(new_config)
            notify("VibeMic", "Settings saved!")
            self._ok()

        elif self.path == "/history/delete":
            try:
                data = json.loads(body)
                delete_history_entry(int(data["index"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                self.send_error(400)
                return
            self._ok()

        elif self.path == "/history/clear":
            clear_history()
            self._ok()

        else:
            self.send_error(404)

    def _ok(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_settings_server = None


def _ensure_server(on_save=None):
    """Start the HTTP server if not already running."""
    global _settings_server, SETTINGS_PORT
    if on_save:
        _SettingsHandler.on_save_callback = on_save
    if _settings_server is None:
        SETTINGS_PORT = _find_free_port()
        _settings_server = http.server.HTTPServer(("127.0.0.1", SETTINGS_PORT), _SettingsHandler)
        t = threading.Thread(target=_settings_server.serve_forever, daemon=True)
        t.start()


def open_settings(on_save):
    """Start settings HTTP server (if not running) and open browser."""
    _ensure_server(on_save)
    webbrowser.open(f"http://127.0.0.1:{SETTINGS_PORT}")


# ─── Recording & transcription ───

def start_recording(tray, update_tray):
    """Start sox recording."""
    global recording_process, is_recording

    if TEMP_WAV.exists():
        TEMP_WAV.unlink()

    try:
        recording_process = subprocess.Popen(
            ["sox", "-d", "-r", "16000", "-c", "1", "-b", "16", str(TEMP_WAV)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        notify("VibeMic", "sox not found. Install: sudo apt install sox", "dialog-error")
        return

    is_recording = True
    update_tray("recording")
    notify("VibeMic", "🎤 Recording... Press PgDn to stop")


def stop_and_transcribe(tray, update_tray):
    """Stop recording, send to Whisper, type the result."""
    global recording_process, is_recording, config

    if not recording_process:
        is_recording = False
        update_tray("idle")
        return

    # Stop sox gracefully
    recording_process.send_signal(signal.SIGINT)
    try:
        recording_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        recording_process.kill()
        recording_process.wait()

    recording_process = None
    is_recording = False
    update_tray("transcribing")
    notify("VibeMic", "⏳ Transcribing...")

    # Validate audio
    if not TEMP_WAV.exists():
        notify("VibeMic", "No audio recorded. Check mic.", "dialog-error")
        update_tray("idle")
        return

    if TEMP_WAV.stat().st_size < MIN_FILE_SIZE:
        notify("VibeMic", "Too short, try again.", "dialog-warning")
        update_tray("idle")
        return

    # Reload config in case settings changed
    config = load_config()
    api_key = config.get("api_key", "")

    if not api_key:
        notify("VibeMic", "No API key. Open Settings to set one.", "dialog-error")
        update_tray("idle")
        return

    try:
        client = OpenAI(api_key=api_key)
        with open(TEMP_WAV, "rb") as f:
            params = {
                "file": f,
                "model": config.get("model", "whisper-1"),
            }
            lang = config.get("language", "")
            if lang:
                params["language"] = lang

            prompt = config.get("prompt", "")
            if prompt:
                params["prompt"] = prompt

            temp = config.get("temperature", 0)
            if temp > 0:
                params["temperature"] = temp

            resp_fmt = config.get("response_format", "json")
            if resp_fmt and resp_fmt != "json":
                params["response_format"] = resp_fmt

            transcription = client.audio.transcriptions.create(**params)

        text = (transcription.text or "").strip()
        if not text:
            notify("VibeMic", "No speech detected.", "dialog-warning")
            update_tray("idle")
            return

        # Save to history
        save_to_history(text)

        # Paste text into the focused window via clipboard (instant, like the Chrome extension)
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE
        )
        proc.communicate(text.encode("utf-8"))
        import time
        time.sleep(0.05)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], timeout=5)
        notify("VibeMic", f"✅ Typed: {text[:60]}{'…' if len(text) > 60 else ''}")
        update_tray("idle")

    except Exception as e:
        msg = str(e)
        if "401" in msg or "Incorrect API key" in msg:
            notify("VibeMic", "Invalid API key. Check Settings.", "dialog-error")
        elif "ENOTFOUND" in msg or "ECONNREFUSED" in msg:
            notify("VibeMic", "Can't reach OpenAI.", "dialog-error")
        else:
            notify("VibeMic", f"Error: {msg[:80]}", "dialog-error")
        update_tray("idle")

    # Cleanup
    try:
        if TEMP_WAV.exists():
            TEMP_WAV.unlink()
    except OSError:
        pass


def on_hotkey(tray, update_tray):
    """Toggle recording on hotkey press."""
    with state_lock:
        if is_recording:
            threading.Thread(target=stop_and_transcribe, args=(tray, update_tray), daemon=True).start()
        else:
            start_recording(tray, update_tray)


def main():
    global config

    import pystray

    config = load_config()

    if not config.get("api_key"):
        print("WARNING: No OpenAI API key found. Open Settings from the tray icon to set one.")

    # Check sox is installed
    if not any((Path(d) / "sox").exists() for d in os.environ.get("PATH", "").split(":")):
        print("ERROR: sox not found. Install: sudo apt install sox libsox-fmt-all")
        sys.exit(1)

    # Tray icon states
    icons = {
        "idle": create_tray_icon((80, 140, 220, 255)),        # Blue
        "recording": create_tray_icon((220, 40, 40, 255)),    # Red
        "transcribing": create_tray_icon((220, 160, 0, 255)), # Orange
    }

    tray = pystray.Icon("vibemic")
    tray.icon = icons["idle"]
    tray.title = "VibeMic — Press PgDn to record"

    def update_tray(state):
        tray.icon = icons.get(state, icons["idle"])
        titles = {
            "idle": "VibeMic — Press PgDn to record",
            "recording": "VibeMic — Recording... PgDn to stop",
            "transcribing": "VibeMic — Transcribing...",
        }
        tray.title = titles.get(state, titles["idle"])

    def on_settings_save(new_config):
        global config
        config = new_config

    def open_settings_click(icon, item):
        open_settings(on_settings_save)

    def open_history_click(icon, item):
        _ensure_server(on_settings_save)
        webbrowser.open(f"http://127.0.0.1:{SETTINGS_PORT}/history")

    def quit_app(icon, item):
        global recording_process
        if recording_process:
            recording_process.kill()
        icon.stop()

    tray.menu = pystray.Menu(
        pystray.MenuItem("VibeMic", None, enabled=False),
        pystray.MenuItem("📋  History", open_history_click),
        pystray.MenuItem("⚙️  Settings...", open_settings_click),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )

    # Global hotkey listener
    def on_press(key):
        if key == RECORD_KEY:
            on_hotkey(tray, update_tray)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    print("VibeMic Native running. Press PgDn to record. Tray icon active.")
    print(f"Config: model={config.get('model')}, language={config.get('language') or 'auto'}")
    print(f"API key: {'✅ set' if config.get('api_key') else '❌ missing — open Settings'}")
    tray.run()


if __name__ == "__main__":
    main()
