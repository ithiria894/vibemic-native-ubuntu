#!/usr/bin/env python3
"""VibeMic Native — Voice-to-text for Ubuntu. Press PgDn to record, PgDn again to transcribe and type."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

from openai import OpenAI
from PIL import Image, ImageDraw
from pynput import keyboard
try:
    from Xlib import X, XK, display as xdisplay
except ImportError:
    X = None
    XK = None
    xdisplay = None


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    display_name: str
    default_base_url: Optional[str]
    suggested_remote_models: tuple
    uses_remote_api: bool


@dataclass(frozen=True)
class LocalModelPreset:
    id: str
    display_name: str
    filename: str
    download_url: str
    description: str

    @property
    def file_path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def is_installed(self) -> bool:
        return self.file_path.exists()


# ─── Paths ───
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
ENV_FILE = SCRIPT_DIR / ".env"
APP_SUPPORT_DIR = Path.home() / ".local" / "share" / "vibemic"
MODELS_DIR = APP_SUPPORT_DIR / "models"
TEMP_DIR = Path.home() / ".cache" / "vibemic"
HISTORY_FILE = SCRIPT_DIR / "history.json"
TEMP_WAV = TEMP_DIR / "recording.wav"
MIN_FILE_SIZE = 1000  # bytes — smaller means no real audio
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
CUSTOM_LOCAL_MODEL_TITLE = "Custom path"

APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ─── Providers & models ───
TRANSCRIPTION_PROVIDERS = [
    ProviderSpec(
        key="openai",
        display_name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        suggested_remote_models=("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"),
        uses_remote_api=True,
    ),
    ProviderSpec(
        key="groq",
        display_name="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        suggested_remote_models=("whisper-large-v3-turbo", "whisper-large-v3"),
        uses_remote_api=True,
    ),
    ProviderSpec(
        key="litellm",
        display_name="LiteLLM",
        default_base_url="http://127.0.0.1:4000/v1",
        suggested_remote_models=("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-large-v3-turbo"),
        uses_remote_api=True,
    ),
    ProviderSpec(
        key="custom-openai-compatible",
        display_name="Custom OpenAI-compatible",
        default_base_url=None,
        suggested_remote_models=("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1", "whisper-large-v3-turbo"),
        uses_remote_api=True,
    ),
    ProviderSpec(
        key="local-whisper-cpp",
        display_name="Local whisper.cpp",
        default_base_url=None,
        suggested_remote_models=(),
        uses_remote_api=False,
    ),
]

PROVIDER_BY_KEY = {provider.key: provider for provider in TRANSCRIPTION_PROVIDERS}
PROVIDER_KEY_BY_DISPLAY = {provider.display_name: provider.key for provider in TRANSCRIPTION_PROVIDERS}

LOCAL_MODEL_PRESETS = [
    LocalModelPreset(
        id="general-balanced",
        display_name="Recommended - Large v3 Q5",
        filename="ggml-large-v3-q5_0.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
        description="Best balance for Cantonese + English mixed speech.",
    ),
    LocalModelPreset(
        id="general-fast-quantized",
        display_name="Fast - Large v3 Turbo Q5",
        filename="ggml-large-v3-turbo-q5_0.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        description="Fastest daily-driver preset with strong multilingual quality.",
    ),
    LocalModelPreset(
        id="general-fast-full",
        display_name="Fast Full - Large v3 Turbo",
        filename="ggml-large-v3-turbo.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        description="Turbo checkpoint in full precision.",
    ),
    LocalModelPreset(
        id="general-max",
        display_name="Max - Large v3",
        filename="ggml-large-v3.bin",
        download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        description="Highest quality base Whisper v3 checkpoint.",
    ),
    LocalModelPreset(
        id="cantonese-focused",
        display_name="Cantonese Focus - Large v3 Cantonese Q8",
        filename="whisper-large-v3-cantonese.q8_0.bin",
        download_url="https://huggingface.co/kiuckhuang/whisper-large-v3-cantonese-ggml/resolve/main/whisper-large-v3-cantonese.q8_0.bin",
        description="Fine-tuned for Cantonese, but can be weaker on English code-switch.",
    ),
    LocalModelPreset(
        id="cantonese-max",
        display_name="Cantonese Max - Large v3 Cantonese BF16",
        filename="whisper-large-v3-cantonese.bf16.bin",
        download_url="https://huggingface.co/kiuckhuang/whisper-large-v3-cantonese-ggml/resolve/main/whisper-large-v3-cantonese.bf16.bin",
        description="Highest-quality Cantonese-specialized checkpoint.",
    ),
]

LOCAL_MODEL_PRESET_BY_ID = {preset.id: preset for preset in LOCAL_MODEL_PRESETS}

CHAT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
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


# ─── Hotkey key-name → X11 keysym mapping ───
SUPPORTED_HOTKEY_NAMES = {
    "page_down",
    "page_up",
    "home",
    "end",
    "insert",
    "delete",
    "scroll_lock",
    "pause",
    "print_screen",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
}

KEY_NAME_TO_XK = {}
if XK is not None:
    KEY_NAME_TO_XK = {
        "page_down": XK.XK_Next,
        "page_up": XK.XK_Prior,
        "home": XK.XK_Home,
        "end": XK.XK_End,
        "insert": XK.XK_Insert,
        "delete": XK.XK_Delete,
        "scroll_lock": XK.XK_Scroll_Lock,
        "pause": XK.XK_Pause,
        "print_screen": XK.XK_Print,
        "f1": XK.XK_F1, "f2": XK.XK_F2, "f3": XK.XK_F3, "f4": XK.XK_F4,
        "f5": XK.XK_F5, "f6": XK.XK_F6, "f7": XK.XK_F7, "f8": XK.XK_F8,
        "f9": XK.XK_F9, "f10": XK.XK_F10, "f11": XK.XK_F11, "f12": XK.XK_F12,
    }


def detect_local_whisper_binary():
    """Find a usable whisper.cpp CLI binary."""
    candidates = [
        os.environ.get("VIBEMIC_WHISPER_CLI"),
        shutil.which("whisper-cli"),
        str(SCRIPT_DIR / "bin" / "whisper-cli"),
        str(Path.home() / ".local" / "bin" / "whisper-cli"),
        "/usr/local/bin/whisper-cli",
        "/usr/bin/whisper-cli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if shutil.which(expanded):
            return shutil.which(expanded)
        if Path(expanded).exists() and os.access(expanded, os.X_OK):
            return expanded
    return ""


# ─── Config management ───
DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-4o-transcribe",
    "transcription_provider": "openai",
    "transcription_base_url": "",
    "transcription_api_key": "",
    "local_whisper_binary_path": detect_local_whisper_binary() or "whisper-cli",
    "local_whisper_model_path": str(LOCAL_MODEL_PRESETS[0].file_path),
    "language": "",
    "prompt": "廣東話、English、普通話、日本語",
    "temperature": 0,
    "response_format": "json",
    "hotkey": "page_down",
    "paraphrase_enabled": False,
    "paraphrase_prompt": (
        "Rewrite this voice transcript into natural work English for Slack or work chat.\n"
        "\n"
        "The input may be in Cantonese, mixed Cantonese/English, or English. Always output in English.\n"
        "\n"
        "Style:\n"
        "- Reads like a real engineer typed it quickly but clearly\n"
        "- Simple, clear, everyday work language — not corporate, not formal\n"
        "- Write like a smart non-native English speaker in tech — natural but not overly polished\n"
        "- Keep the original meaning and technical terms accurate\n"
        "- Fix rough or broken language naturally — do not over-fix\n"
        "- Same length or shorter. Do not add context that was not in the original\n"
        "- Preserve the original intent, including uncertainty, directness, or casual tone\n"
        "- Natural flow, slightly uneven sentences are fine\n"
        "- No em dash\n"
        "\n"
        "Never use: just a quick update, for your reference, I wanted to let you know, "
        "please let me know if you have any questions, moving forward, aligned on, "
        "happy to, sounds good, on my side\n"
        "\n"
        "Return only the rewritten text. No explanation."
    ),
    "paraphrase_model": "gpt-4o-mini",
}


def normalize_config(raw_config):
    """Merge config with defaults and coerce new keys for older installs."""
    normalized = dict(DEFAULT_CONFIG)
    normalized.update(raw_config or {})

    provider_key = normalized.get("transcription_provider", "openai")
    if provider_key not in PROVIDER_BY_KEY:
        provider_key = "openai"
    normalized["transcription_provider"] = provider_key

    if not normalized.get("local_whisper_binary_path"):
        normalized["local_whisper_binary_path"] = detect_local_whisper_binary() or "whisper-cli"

    if not normalized.get("local_whisper_model_path"):
        normalized["local_whisper_model_path"] = str(LOCAL_MODEL_PRESETS[0].file_path)

    if normalized.get("response_format") not in RESPONSE_FORMATS:
        normalized["response_format"] = "json"

    if normalized.get("hotkey") not in SUPPORTED_HOTKEY_NAMES:
        normalized["hotkey"] = "page_down"

    return normalized


def load_config():
    """Load config from config.json, falling back to .env for API key."""
    config = dict(DEFAULT_CONFIG)

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    config = normalize_config(config)

    if not config.get("api_key"):
        config["api_key"] = _load_env_api_key()

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
    normalized = normalize_config(config)
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Failed to save config: {e}")


def provider_spec_for_config(config):
    """Resolve provider metadata from config."""
    return PROVIDER_BY_KEY.get(config.get("transcription_provider"), PROVIDER_BY_KEY["openai"])


def provider_spec_from_display(display_name):
    """Resolve provider metadata from the UI display label."""
    key = PROVIDER_KEY_BY_DISPLAY.get(display_name, "openai")
    return PROVIDER_BY_KEY[key]


def resolved_transcription_base_url(config):
    """Provider base URL, falling back to the provider default when unset."""
    configured = str(config.get("transcription_base_url", "")).strip()
    if configured:
        return configured
    provider = provider_spec_for_config(config)
    return provider.default_base_url or ""


def resolved_transcription_api_key(config):
    """Provider-specific key override, falling back to the default API key."""
    override = str(config.get("transcription_api_key", "")).strip()
    if override:
        return override
    return str(config.get("api_key", "")).strip()


def resolve_binary_path(path_value):
    """Resolve a binary path or bare executable name into something runnable."""
    candidate = str(path_value or "").strip()
    if not candidate:
        candidate = detect_local_whisper_binary() or "whisper-cli"

    expanded = os.path.expanduser(candidate)
    if os.path.sep not in candidate and shutil.which(candidate):
        return shutil.which(candidate)
    if shutil.which(expanded):
        return shutil.which(expanded)
    return expanded


def resolved_local_whisper_binary_path(config):
    """Return the best current whisper.cpp binary path."""
    return resolve_binary_path(config.get("local_whisper_binary_path", ""))


def resolved_local_whisper_model_path(config):
    """Return the current local model path with ~ expanded."""
    configured = str(config.get("local_whisper_model_path", "")).strip()
    if not configured:
        configured = str(LOCAL_MODEL_PRESETS[0].file_path)
    return str(Path(configured).expanduser())


def preset_for_path(path_value):
    """Find a known local model preset that matches the current path."""
    normalized = str(Path(path_value).expanduser())
    for preset in LOCAL_MODEL_PRESETS:
        if str(preset.file_path) == normalized:
            return preset
    return None


def transcription_readiness_issue(config):
    """Human-readable validation for the selected provider."""
    provider = provider_spec_for_config(config)

    if provider.key == "local-whisper-cpp":
        binary_path = resolved_local_whisper_binary_path(config)
        if not binary_path or not Path(binary_path).exists() or not os.access(binary_path, os.X_OK):
            return "Local whisper.cpp binary not found. Open Settings and point Local Binary Path to whisper-cli."

        model_path = Path(resolved_local_whisper_model_path(config))
        if not model_path.exists():
            return "Local Whisper model not found. Download a model from Settings first."

        return None

    if not resolved_transcription_base_url(config):
        return "No transcription base URL. Open Settings to configure the provider."

    if not resolved_transcription_api_key(config):
        return "No transcription API key. Open Settings to configure the provider."

    return None


def format_bytes(num_bytes):
    """Compact human-readable byte display."""
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024


def extract_transcription_text(response):
    """Handle the SDK's different response shapes."""
    if response is None:
        return ""

    if hasattr(response, "text") and response.text is not None:
        return str(response.text).strip()

    if isinstance(response, dict):
        return str(response.get("text", "")).strip()

    if isinstance(response, str):
        return response.strip()

    return str(response).strip()


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


def save_to_history(text, original=None):
    """Append a transcript to history. If paraphrased, pass original transcript too."""
    from datetime import datetime

    history = load_history()
    entry = {
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if original and original != text:
        entry["original"] = original
    history.insert(0, entry)
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
    size = 48
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    mic_color = (255, 255, 255, 240)

    draw.ellipse([0, 0, size - 1, size - 1], fill=color)
    draw.rounded_rectangle([17, 7, 31, 27], radius=7, fill=mic_color)
    draw.arc([11, 16, 37, 38], 0, 180, fill=mic_color, width=3)
    draw.line([24, 37, 24, 42], fill=mic_color, width=3)
    draw.line([18, 42, 30, 42], fill=mic_color, width=3)

    return img


# ─── Theme constants ───
BG = "#1a1a2e"
FG = "#e0e0e0"
ACCENT = "#64b5f6"
INPUT_BG = "#16213e"
BORDER = "#2a2a4a"
SUBTLE = "#888888"
SUCCESS = "#7bd88f"
WARNING = "#ffb86c"
ERROR = "#ff6b6b"


def _apply_theme(root):
    """Apply dark theme to a tkinter root/toplevel."""
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TLabel", background=BG, foreground=FG, font=("sans-serif", 10))
    style.configure(
        "TButton",
        background=INPUT_BG,
        foreground=FG,
        bordercolor=BORDER,
        relief="flat",
        padding=(10, 6),
    )
    style.map("TButton", background=[("active", "#252550")])
    style.configure("Accent.TButton", background=ACCENT, foreground=BG, bordercolor=ACCENT, relief="flat", padding=(10, 6))
    style.map("Accent.TButton", background=[("active", "#90caf9")])
    style.configure(
        "TEntry",
        fieldbackground=INPUT_BG,
        foreground=FG,
        bordercolor=BORDER,
        insertcolor=FG,
    )
    style.configure(
        "TCombobox",
        fieldbackground=INPUT_BG,
        foreground=FG,
        background=INPUT_BG,
        selectbackground=INPUT_BG,
        selectforeground=ACCENT,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", INPUT_BG)],
        foreground=[("readonly", FG)],
        selectbackground=[("readonly", INPUT_BG)],
    )
    style.configure("TScale", background=BG, troughcolor=INPUT_BG, slidercolor=ACCENT)
    style.configure("TCheckbutton", background=BG, foreground=FG, focuscolor="", indicatorcolor=INPUT_BG)
    style.map("TCheckbutton", background=[("active", BG)], indicatorcolor=[("selected", ACCENT)])
    style.configure("TScrollbar", background=INPUT_BG, troughcolor=BG, bordercolor=BG, arrowcolor=FG)


def _label(parent, text, size=10, bold=False, color=None):
    font = ("sans-serif", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, bg=BG, fg=color or FG, font=font)


def _text_widget(parent, height=2):
    return tk.Text(
        parent,
        height=height,
        bg=INPUT_BG,
        fg=FG,
        insertbackground=FG,
        relief="flat",
        bd=1,
        font=("sans-serif", 10),
        padx=8,
        pady=6,
        wrap="word",
        highlightbackground=BORDER,
        highlightthickness=1,
    )


class TranscriberError(RuntimeError):
    """Domain-specific transcription failure."""


class TranscriptionProvider:
    """Simple provider protocol."""

    def transcribe(self, file_path, current_config):
        raise NotImplementedError


class OpenAICompatibleTranscriptionProvider(TranscriptionProvider):
    """Any provider that speaks the OpenAI audio transcription API."""

    def transcribe(self, file_path, current_config):
        api_key = resolved_transcription_api_key(current_config)
        base_url = resolved_transcription_base_url(current_config)

        if not api_key:
            raise TranscriberError("No transcription API key configured.")
        if not base_url:
            raise TranscriberError("No transcription base URL configured.")

        client = OpenAI(api_key=api_key, base_url=base_url)

        with open(file_path, "rb") as audio_file:
            params = {
                "file": audio_file,
                "model": current_config.get("model", "gpt-4o-transcribe").strip() or "gpt-4o-transcribe",
                "response_format": current_config.get("response_format", "json"),
            }

            language = current_config.get("language", "").strip()
            if language:
                params["language"] = language

            prompt = current_config.get("prompt", "").strip()
            if prompt:
                params["prompt"] = prompt

            temperature = float(current_config.get("temperature", 0) or 0)
            if temperature > 0:
                params["temperature"] = temperature

            transcription = client.audio.transcriptions.create(**params)

        text = extract_transcription_text(transcription)
        if not text:
            raise TranscriberError("No speech detected.")
        return text


class LocalWhisperCppTranscriptionProvider(TranscriptionProvider):
    """Local whisper.cpp provider that shells out to whisper-cli."""

    def transcribe(self, file_path, current_config):
        binary_path = resolved_local_whisper_binary_path(current_config)
        model_path = resolved_local_whisper_model_path(current_config)

        if not binary_path or not Path(binary_path).exists() or not os.access(binary_path, os.X_OK):
            raise TranscriberError("Local whisper.cpp binary not found.")

        if not Path(model_path).exists():
            raise TranscriberError("Local Whisper model not found.")

        with tempfile.TemporaryDirectory(prefix="vibemic-whisper-") as temp_dir:
            output_base = Path(temp_dir) / "transcript"
            command = [
                binary_path,
                "-m", model_path,
                "-f", str(file_path),
                "-otxt",
                "-of", str(output_base),
                "-np",
                "-nt",
                "-l", current_config.get("language", "").strip() or "auto",
            ]

            prompt = current_config.get("prompt", "").strip()
            if prompt:
                command.extend(["--prompt", prompt])

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "Local whisper.cpp failed.").strip()
                raise TranscriberError(detail)

            transcript_path = output_base.with_suffix(".txt")
            if not transcript_path.exists():
                detail = (result.stderr or result.stdout or "Local whisper.cpp did not produce a transcript.").strip()
                raise TranscriberError(detail)

            text = transcript_path.read_text(encoding="utf-8").strip()
            if not text:
                raise TranscriberError("No speech detected.")

            return text


def make_transcription_provider(current_config):
    """Pick the configured transcription backend."""
    provider = provider_spec_for_config(current_config)
    if provider.key == "local-whisper-cpp":
        return LocalWhisperCppTranscriptionProvider()
    return OpenAICompatibleTranscriptionProvider()


# ─── Native Settings Dialog ───
def open_settings_dialog(on_save=None, on_hotkey_change=None):
    """Open native tkinter settings window."""

    def run():
        cfg = load_config()

        root = tk.Tk()
        root.title("VibeMic Settings")
        root.configure(bg=BG)
        root.resizable(False, False)
        _apply_theme(root)

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, width=620)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=BG, padx=24, pady=20)
        win_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_frame_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)

        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def mousewheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", mousewheel)
        canvas.bind_all("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))

        _label(frame, "VibeMic Settings", size=16, bold=True, color=ACCENT).pack(anchor="w", pady=(0, 16))

        def make_block(title, hint_text=None):
            block = tk.Frame(frame, bg=BG)
            block.pack(fill="x", pady=(12, 0))
            _label(block, title, size=10, bold=True).pack(anchor="w")
            if hint_text:
                tk.Label(
                    block,
                    text=hint_text,
                    bg=BG,
                    fg=SUBTLE,
                    font=("sans-serif", 9),
                    justify="left",
                    wraplength=560,
                ).pack(anchor="w", pady=(2, 0))
            return block

        def divider():
            tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(18, 6))

        def make_entry_row(parent, text_var, show=None, width=44):
            row = tk.Frame(parent, bg=BG)
            row.pack(fill="x", pady=(4, 0))
            entry = ttk.Entry(row, textvariable=text_var, show=show, width=width)
            entry.pack(side="left", fill="x", expand=True)
            return row, entry

        def toggle_mask(entry_widget):
            entry_widget.config(show="" if entry_widget.cget("show") else "•")

        api_var = tk.StringVar(value=cfg.get("api_key", ""))
        provider_var = tk.StringVar(value=provider_spec_for_config(cfg).display_name)
        model_var = tk.StringVar(value=cfg.get("model", "gpt-4o-transcribe"))
        trans_api_var = tk.StringVar(value=cfg.get("transcription_api_key", ""))
        base_url_var = tk.StringVar(value=cfg.get("transcription_base_url", ""))
        local_binary_var = tk.StringVar(value=cfg.get("local_whisper_binary_path", detect_local_whisper_binary() or "whisper-cli"))
        local_model_path_var = tk.StringVar(value=resolved_local_whisper_model_path(cfg))
        local_preset_var = tk.StringVar(value="")
        lang_names = [name for name, _ in LANGUAGES]
        lang_codes = [code for _, code in LANGUAGES]
        current_code = cfg.get("language", "")
        current_lang_index = lang_codes.index(current_code) if current_code in lang_codes else 0
        lang_var = tk.StringVar(value=lang_names[current_lang_index])
        temp_var = tk.DoubleVar(value=cfg.get("temperature", 0))
        fmt_var = tk.StringVar(value=cfg.get("response_format", "json"))
        hotkey_var = tk.StringVar(value=cfg.get("hotkey", "page_down"))
        para_enabled_var = tk.BooleanVar(value=cfg.get("paraphrase_enabled", False))
        para_model_var = tk.StringVar(value=cfg.get("paraphrase_model", "gpt-4o-mini"))
        local_model_status_var = tk.StringVar(value="")
        remote_model_hint_var = tk.StringVar(value="")
        base_url_hint_var = tk.StringVar(value="")
        download_state_vars = {}
        download_button_widgets = {}
        download_status_labels = {}
        download_status_overrides = {}
        downloads_in_progress = set()
        rows = {}

        def update_label_color(label_widget, status_kind):
            colors = {
                "ok": SUCCESS,
                "warn": WARNING,
                "error": ERROR,
                "subtle": SUBTLE,
            }
            label_widget.config(fg=colors.get(status_kind, FG))

        api_block = make_block("Default API Key", "Used for remote transcription fallback and for paraphrase mode.")
        api_row, api_entry = make_entry_row(api_block, api_var, show="•")
        ttk.Button(api_row, text="Show", command=lambda: toggle_mask(api_entry)).pack(side="left", padx=(6, 0))

        provider_block = make_block("Transcription Provider")
        provider_combo = ttk.Combobox(
            provider_block,
            textvariable=provider_var,
            values=[provider.display_name for provider in TRANSCRIPTION_PROVIDERS],
            state="readonly",
            width=42,
        )
        provider_combo.pack(fill="x", pady=(4, 0))

        rows["remote_model"] = make_block("Remote Model")
        model_combo = ttk.Combobox(rows["remote_model"], textvariable=model_var, values=[], width=42)
        model_combo.pack(fill="x", pady=(4, 0))
        remote_model_hint = tk.Label(rows["remote_model"], textvariable=remote_model_hint_var, bg=BG, fg=SUBTLE, font=("sans-serif", 9), justify="left")
        remote_model_hint.pack(anchor="w", pady=(4, 0))

        rows["transcription_api_key"] = make_block("Transcription API Key", "Optional override for the selected provider. Leave blank to use Default API Key.")
        trans_api_row, trans_api_entry = make_entry_row(rows["transcription_api_key"], trans_api_var, show="•")
        ttk.Button(trans_api_row, text="Show", command=lambda: toggle_mask(trans_api_entry)).pack(side="left", padx=(6, 0))

        rows["transcription_base_url"] = make_block("Base URL")
        base_url_row, _ = make_entry_row(rows["transcription_base_url"], base_url_var)
        ttk.Button(base_url_row, text="Reset", command=lambda: base_url_var.set("")).pack(side="left", padx=(6, 0))
        base_url_hint = tk.Label(rows["transcription_base_url"], textvariable=base_url_hint_var, bg=BG, fg=SUBTLE, font=("sans-serif", 9), justify="left")
        base_url_hint.pack(anchor="w", pady=(4, 0))

        rows["local_binary_path"] = make_block("Local Binary Path", "Point this to `whisper-cli` from whisper.cpp.")
        local_binary_row, _ = make_entry_row(rows["local_binary_path"], local_binary_var)

        def browse_local_binary():
            selected = filedialog.askopenfilename(title="Choose whisper-cli")
            if selected:
                local_binary_var.set(selected)

        ttk.Button(local_binary_row, text="Browse", command=browse_local_binary).pack(side="left", padx=(6, 0))

        rows["local_model_preset"] = make_block("Local Model Preset", "Choose a preset after download, or keep a custom path.")
        local_preset_values = [preset.display_name for preset in LOCAL_MODEL_PRESETS] + [CUSTOM_LOCAL_MODEL_TITLE]
        local_preset_combo = ttk.Combobox(rows["local_model_preset"], textvariable=local_preset_var, values=local_preset_values, state="readonly", width=42)
        local_preset_combo.pack(fill="x", pady=(4, 0))

        rows["local_model_path"] = make_block("Local Model Path")
        local_model_row, _ = make_entry_row(rows["local_model_path"], local_model_path_var)

        def browse_local_model():
            selected = filedialog.askopenfilename(
                title="Choose local Whisper model",
                filetypes=[("Whisper model", "*.bin"), ("All files", "*.*")],
            )
            if selected:
                local_model_path_var.set(selected)
                refresh_local_model_selection()

        ttk.Button(local_model_row, text="Browse", command=browse_local_model).pack(side="left", padx=(6, 0))
        local_model_status = tk.Label(rows["local_model_path"], textvariable=local_model_status_var, bg=BG, fg=SUBTLE, font=("sans-serif", 9), justify="left")
        local_model_status.pack(anchor="w", pady=(4, 0))

        rows["model_library"] = make_block("Model Library", "Each preset needs to be downloaded once before it can be used.")
        library_header = tk.Frame(rows["model_library"], bg=BG)
        library_header.pack(fill="x", pady=(4, 8))

        def open_models_folder():
            folder = str(MODELS_DIR)
            opener = shutil.which("xdg-open")
            if opener:
                subprocess.Popen([opener, folder])
            else:
                messagebox.showinfo("VibeMic", f"Models folder:\n{folder}", parent=root)

        ttk.Button(library_header, text="Open Folder", command=open_models_folder).pack(side="right")
        model_library_rows = tk.Frame(rows["model_library"], bg=BG)
        model_library_rows.pack(fill="x")

        rows["language"] = make_block("Language")
        ttk.Combobox(rows["language"], textvariable=lang_var, values=lang_names, state="readonly", width=42).pack(fill="x", pady=(4, 0))

        rows["prompt"] = make_block("Transcription Prompt", "Hint for the speech recognizer — expected languages or vocabulary.")
        prompt_text = _text_widget(rows["prompt"], height=2)
        prompt_text.insert("1.0", cfg.get("prompt", ""))
        prompt_text.pack(fill="x", pady=(4, 0))

        rows["temperature"] = make_block("Temperature")
        temp_frame = tk.Frame(rows["temperature"], bg=BG)
        temp_frame.pack(fill="x", pady=(4, 0))
        temp_value_label = tk.Label(temp_frame, text=f"{temp_var.get():.1f}", bg=BG, fg=ACCENT, font=("sans-serif", 10, "bold"), width=4)
        temp_value_label.pack(side="right")

        def on_temp(value):
            temp_value_label.config(text=f"{float(value):.1f}")

        ttk.Scale(temp_frame, from_=0, to=1, variable=temp_var, command=on_temp, orient="horizontal").pack(side="left", fill="x", expand=True)

        rows["response_format"] = make_block("Response Format")
        ttk.Combobox(rows["response_format"], textvariable=fmt_var, values=RESPONSE_FORMATS, state="readonly", width=42).pack(fill="x", pady=(4, 0))

        rows["hotkey"] = make_block("Record Hotkey", "Click Change, then press a special key such as PgDn, F-key, Home, or End.")
        hotkey_frame = tk.Frame(rows["hotkey"], bg=BG)
        hotkey_frame.pack(fill="x", pady=(4, 0))
        hotkey_display = ttk.Entry(hotkey_frame, textvariable=hotkey_var, state="readonly", width=20)
        hotkey_display.pack(side="left")
        capturing = [False]

        def start_capture():
            if capturing[0]:
                return

            capturing[0] = True
            change_btn.config(text="Press a key...", state="disabled")

            def on_key(key):
                if not capturing[0]:
                    return False
                key_name = getattr(key, "name", None)
                if key_name and key_name in SUPPORTED_HOTKEY_NAMES:
                    hotkey_var.set(key_name)
                    capturing[0] = False
                    root.after(0, lambda: change_btn.config(text="Change", state="normal"))
                    return False
                return None

            listener = keyboard.Listener(on_press=on_key)
            listener.daemon = True
            listener.start()

        change_btn = ttk.Button(hotkey_frame, text="Change", command=start_capture)
        change_btn.pack(side="left", padx=(8, 0))

        divider()
        _label(frame, "Paraphrase", size=13, bold=True, color=ACCENT).pack(anchor="w", pady=(4, 0))
        tk.Label(
            frame,
            text="After transcription, rewrite text with an AI prompt before typing. This still uses the Default API Key.",
            bg=BG,
            fg=SUBTLE,
            font=("sans-serif", 9),
            justify="left",
            wraplength=560,
        ).pack(anchor="w", pady=(2, 0))

        ttk.Checkbutton(frame, text="Enable paraphrase mode", variable=para_enabled_var).pack(anchor="w", pady=(8, 0))

        para_prompt_block = make_block("Paraphrase Prompt")
        para_prompt_text = _text_widget(para_prompt_block, height=4)
        para_prompt_text.insert("1.0", cfg.get("paraphrase_prompt", DEFAULT_CONFIG["paraphrase_prompt"]))
        para_prompt_text.pack(fill="x", pady=(4, 0))

        para_model_block = make_block("Paraphrase Model")
        ttk.Combobox(para_model_block, textvariable=para_model_var, values=CHAT_MODELS, state="readonly", width=42).pack(fill="x", pady=(4, 0))

        def refresh_remote_model_suggestions():
            provider = provider_spec_from_display(provider_var.get())
            values = list(provider.suggested_remote_models)
            model_combo["values"] = values
            if provider.uses_remote_api and values and model_var.get().strip() not in values:
                model_var.set(values[0])
            remote_model_hint_var.set(
                "Suggested: " + ", ".join(values) if values else "Custom providers can use any model string their endpoint supports."
            )
            if provider.default_base_url:
                base_url_hint_var.set(f"Leave blank to use default: {provider.default_base_url}")
            elif provider.uses_remote_api:
                base_url_hint_var.set("Required for custom OpenAI-compatible endpoints.")
            else:
                base_url_hint_var.set("Not used for local transcription.")

        def refresh_local_model_selection():
            selected_path = resolved_local_whisper_model_path({"local_whisper_model_path": local_model_path_var.get()})
            matched_preset = preset_for_path(selected_path)
            if matched_preset:
                local_preset_var.set(matched_preset.display_name)
                if matched_preset.is_installed:
                    local_model_status_var.set(f"Installed: {matched_preset.filename}")
                    update_label_color(local_model_status, "ok")
                else:
                    local_model_status_var.set(f"Download required: {matched_preset.filename}")
                    update_label_color(local_model_status, "warn")
            else:
                local_preset_var.set(CUSTOM_LOCAL_MODEL_TITLE)
                model_path = Path(selected_path)
                if model_path.exists():
                    local_model_status_var.set(f"Custom model found: {model_path.name}")
                    update_label_color(local_model_status, "ok")
                else:
                    local_model_status_var.set("Custom path is missing or not downloaded yet.")
                    update_label_color(local_model_status, "warn")

        def select_local_model_preset(event=None):
            del event
            selected_display = local_preset_var.get()
            if selected_display == CUSTOM_LOCAL_MODEL_TITLE:
                refresh_local_model_selection()
                return
            preset = next((item for item in LOCAL_MODEL_PRESETS if item.display_name == selected_display), None)
            if preset:
                local_model_path_var.set(str(preset.file_path))
            refresh_local_model_selection()
            refresh_model_library_rows()

        local_preset_combo.bind("<<ComboboxSelected>>", select_local_model_preset)

        def refresh_model_library_rows():
            current_selected = preset_for_path(local_model_path_var.get())
            for preset in LOCAL_MODEL_PRESET_BY_ID.values():
                installed = preset.is_installed
                in_progress = preset.id in downloads_in_progress
                status_var = download_state_vars[preset.id]
                button_widget = download_button_widgets[preset.id]
                status_label = download_status_labels[preset.id]

                if in_progress:
                    status_text = download_status_overrides.get(preset.id, "Downloading...")
                    button_widget.config(text="Downloading...", state="disabled")
                    update_label_color(status_label, "warn")
                elif installed:
                    status_text = "Installed"
                    if current_selected and current_selected.id == preset.id:
                        status_text = "Installed • active"
                    button_widget.config(text="Use", state="normal")
                    update_label_color(status_label, "ok")
                else:
                    status_text = "Not downloaded"
                    button_widget.config(text="Download", state="normal")
                    update_label_color(status_label, "subtle")

                if preset.id in download_status_overrides and not in_progress:
                    status_text = download_status_overrides[preset.id]
                status_var.set(status_text)

        def make_library_action(preset):
            def action():
                if preset.is_installed and preset.id not in downloads_in_progress:
                    local_model_path_var.set(str(preset.file_path))
                    local_preset_var.set(preset.display_name)
                    refresh_local_model_selection()
                    refresh_model_library_rows()
                    return

                if preset.id in downloads_in_progress:
                    return

                downloads_in_progress.add(preset.id)
                download_status_overrides[preset.id] = "Preparing..."
                refresh_model_library_rows()

                def set_progress(message):
                    download_status_overrides[preset.id] = message
                    refresh_model_library_rows()

                def finalize_success():
                    downloads_in_progress.discard(preset.id)
                    download_status_overrides.pop(preset.id, None)
                    if local_preset_var.get() == preset.display_name or not Path(local_model_path_var.get()).exists():
                        local_model_path_var.set(str(preset.file_path))
                        local_preset_var.set(preset.display_name)
                    refresh_local_model_selection()
                    refresh_model_library_rows()
                    messagebox.showinfo("VibeMic", f"Downloaded {preset.display_name}", parent=root)

                def finalize_failure(error_text):
                    downloads_in_progress.discard(preset.id)
                    download_status_overrides[preset.id] = f"Failed: {error_text}"
                    refresh_model_library_rows()
                    messagebox.showerror("VibeMic", f"Failed to download {preset.display_name}\n\n{error_text}", parent=root)

                def worker():
                    temp_path = preset.file_path.with_suffix(preset.file_path.suffix + ".part")
                    try:
                        MODELS_DIR.mkdir(parents=True, exist_ok=True)
                        request = urllib.request.Request(
                            preset.download_url,
                            headers={"User-Agent": "VibeMic/1.0"},
                        )
                        with urllib.request.urlopen(request, timeout=60) as response, open(temp_path, "wb") as temp_file:
                            total = int(response.headers.get("Content-Length", "0"))
                            downloaded = 0
                            while True:
                                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                                if not chunk:
                                    break
                                temp_file.write(chunk)
                                downloaded += len(chunk)
                                if total > 0:
                                    percent = downloaded / total * 100
                                    progress = f"Downloading {percent:.0f}% ({format_bytes(downloaded)} / {format_bytes(total)})"
                                else:
                                    progress = f"Downloading {format_bytes(downloaded)}"
                                root.after(0, set_progress, progress)

                        temp_path.replace(preset.file_path)
                        root.after(0, finalize_success)
                    except Exception as exc:
                        try:
                            if temp_path.exists():
                                temp_path.unlink()
                        except OSError:
                            pass
                        root.after(0, finalize_failure, str(exc))

                threading.Thread(target=worker, daemon=True).start()

            return action

        for preset in LOCAL_MODEL_PRESETS:
            row = tk.Frame(model_library_rows, bg=INPUT_BG, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=(0, 6))

            info = tk.Frame(row, bg=INPUT_BG)
            info.pack(side="left", fill="x", expand=True)
            tk.Label(info, text=preset.display_name, bg=INPUT_BG, fg=FG, font=("sans-serif", 10, "bold"), anchor="w").pack(anchor="w")
            tk.Label(info, text=preset.description, bg=INPUT_BG, fg=SUBTLE, font=("sans-serif", 9), anchor="w", justify="left", wraplength=330).pack(anchor="w", pady=(2, 0))
            tk.Label(info, text=preset.filename, bg=INPUT_BG, fg=SUBTLE, font=("sans-serif", 8), anchor="w").pack(anchor="w", pady=(2, 0))

            controls = tk.Frame(row, bg=INPUT_BG)
            controls.pack(side="right", anchor="e")

            status_var = tk.StringVar(value="")
            status_label = tk.Label(controls, textvariable=status_var, bg=INPUT_BG, fg=SUBTLE, font=("sans-serif", 9))
            status_label.pack(anchor="e")
            action_button = ttk.Button(controls, text="Download", command=make_library_action(preset))
            action_button.pack(anchor="e", pady=(6, 0))

            download_state_vars[preset.id] = status_var
            download_button_widgets[preset.id] = action_button
            download_status_labels[preset.id] = status_label

        def place_rows_in_order(row_keys, after_widget):
            previous = after_widget
            for key in row_keys:
                rows[key].pack_forget()
                rows[key].pack(fill="x", pady=(12, 0), after=previous)
                previous = rows[key]

        def toggle_provider_rows():
            provider = provider_spec_from_display(provider_var.get())
            is_local = provider.key == "local-whisper-cpp"
            local_keys = ["local_binary_path", "local_model_preset", "local_model_path", "model_library"]
            remote_keys = ["remote_model", "transcription_api_key", "transcription_base_url"]

            for key in local_keys + remote_keys:
                rows[key].pack_forget()

            if is_local:
                place_rows_in_order(local_keys, provider_block)
            else:
                place_rows_in_order(remote_keys, provider_block)

            refresh_remote_model_suggestions()
            refresh_local_model_selection()
            refresh_model_library_rows()

        provider_combo.bind("<<ComboboxSelected>>", lambda event: toggle_provider_rows())
        local_model_path_var.trace_add("write", lambda *args: refresh_local_model_selection())

        selected_preset = preset_for_path(local_model_path_var.get())
        if selected_preset:
            local_preset_var.set(selected_preset.display_name)
        else:
            local_preset_var.set(CUSTOM_LOCAL_MODEL_TITLE)

        refresh_remote_model_suggestions()
        refresh_local_model_selection()
        refresh_model_library_rows()
        toggle_provider_rows()

        tk.Frame(frame, bg=BG, height=12).pack()
        button_row = tk.Frame(frame, bg=BG)
        button_row.pack(fill="x", pady=(8, 4))

        def do_save():
            provider = provider_spec_from_display(provider_var.get())
            language_name = lang_var.get()
            language_code = next((code for name, code in LANGUAGES if name == language_name), "")

            model_name = model_var.get().strip()
            if provider.uses_remote_api and not model_name:
                suggestions = list(provider.suggested_remote_models)
                model_name = suggestions[0] if suggestions else DEFAULT_CONFIG["model"]

            new_cfg = {
                "api_key": api_var.get().strip(),
                "model": model_name or cfg.get("model", DEFAULT_CONFIG["model"]),
                "transcription_provider": provider.key,
                "transcription_base_url": base_url_var.get().strip(),
                "transcription_api_key": trans_api_var.get().strip(),
                "local_whisper_binary_path": local_binary_var.get().strip(),
                "local_whisper_model_path": local_model_path_var.get().strip(),
                "language": language_code,
                "prompt": prompt_text.get("1.0", "end-1c").strip(),
                "temperature": round(temp_var.get(), 1),
                "response_format": fmt_var.get(),
                "hotkey": hotkey_var.get(),
                "paraphrase_enabled": para_enabled_var.get(),
                "paraphrase_prompt": para_prompt_text.get("1.0", "end-1c").strip(),
                "paraphrase_model": para_model_var.get(),
            }

            if new_cfg["hotkey"] not in SUPPORTED_HOTKEY_NAMES:
                messagebox.showerror("VibeMic", "Choose a supported special key for the hotkey.", parent=root)
                return

            readiness_issue = transcription_readiness_issue(new_cfg)
            if readiness_issue:
                messagebox.showerror("VibeMic", readiness_issue, parent=root)
                return

            if new_cfg["paraphrase_enabled"] and not new_cfg["api_key"]:
                messagebox.showerror("VibeMic", "Default API Key is required for paraphrase mode.", parent=root)
                return

            save_config(new_cfg)
            if on_save:
                on_save(new_cfg)
            if on_hotkey_change:
                on_hotkey_change(new_cfg["hotkey"])

            notify("VibeMic", "Settings saved!")
            root.destroy()

        ttk.Button(button_row, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(button_row, text="Save", command=do_save, style="Accent.TButton").pack(side="right")

        root.update_idletasks()
        screen_height = root.winfo_screenheight()
        window_height = min(frame.winfo_reqheight() + 40, int(screen_height * 0.9))
        root.geometry(f"660x{window_height}")
        root.mainloop()

    threading.Thread(target=run, daemon=True).start()


# ─── Native History Dialog ───
def open_history_dialog():
    """Open native tkinter history window."""

    def run():
        root = tk.Tk()
        root.title("VibeMic History")
        root.geometry("640x520")
        root.configure(bg=BG)
        _apply_theme(root)

        header = tk.Frame(root, bg=BG, padx=16, pady=12)
        header.pack(fill="x")
        _label(header, "VibeMic History", size=15, bold=True, color=ACCENT).pack(side="left")

        count_label = tk.Label(header, text="", bg=BG, fg=SUBTLE, font=("sans-serif", 10))
        count_label.pack(side="left", padx=(12, 0))

        list_outer = tk.Frame(root, bg=BG)
        list_outer.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        canvas = tk.Canvas(list_outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_inner_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda event: canvas.yview_scroll(1, "units"))

        card_widgets = []

        def refresh():
            for widget in card_widgets:
                widget.destroy()
            card_widgets.clear()
            history = load_history()
            count_label.config(text=f"{len(history)} transcript{'s' if len(history) != 1 else ''}")

            if not history:
                label = _label(inner, "No transcripts yet. Press PgDn to record!", color=SUBTLE)
                label.pack(pady=40)
                card_widgets.append(label)
                return

            for index, entry in enumerate(history):
                build_card(index, entry.get("text", ""), entry.get("timestamp", ""), entry.get("original"))

        def build_card(index, text, timestamp, original=None):
            card_bg = "#1e2d45" if original else INPUT_BG
            card = tk.Frame(inner, bg=card_bg, padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", pady=(0, 6))
            card_widgets.append(card)

            row = tk.Frame(card, bg=card_bg)
            row.pack(fill="x")

            ts_text = f"✍️ {timestamp}" if original else timestamp
            tk.Label(row, text=ts_text, bg=card_bg, fg=SUBTLE, font=("sans-serif", 9)).pack(side="left")

            def do_delete():
                delete_history_entry(index)
                refresh()

            def do_copy():
                root.clipboard_clear()
                root.clipboard_append(text)
                root.after(3000, root.clipboard_clear)

            ttk.Button(row, text="Delete", command=do_delete).pack(side="right", padx=(4, 0))
            ttk.Button(row, text="Copy", command=do_copy).pack(side="right")

            tk.Label(card, text=text, bg=card_bg, fg=FG, font=("sans-serif", 11), justify="left", wraplength=560, anchor="w").pack(fill="x", pady=(6, 0), anchor="w")

            if original:
                tk.Label(card, text="Original:", bg=card_bg, fg=SUBTLE, font=("sans-serif", 9, "bold")).pack(anchor="w", pady=(8, 0))
                tk.Label(card, text=original, bg=card_bg, fg="#aaaaaa", font=("sans-serif", 10), justify="left", wraplength=560, anchor="w").pack(fill="x", anchor="w")

        def do_clear():
            if messagebox.askyesno("Clear History", "Clear all transcript history?", parent=root):
                clear_history()
                refresh()

        ttk.Button(header, text="Clear All", command=do_clear).pack(side="right")
        refresh()
        root.mainloop()

    threading.Thread(target=run, daemon=True).start()


# ─── Paraphrase ───
def paraphrase_text(text, api_key, para_prompt, model="gpt-4o-mini"):
    """Use OpenAI chat completions to paraphrase the transcript."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": para_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─── Recording & transcription ───
def start_recording(tray, update_tray):
    """Start sox recording."""
    global recording_process, is_recording

    if TEMP_WAV.exists():
        TEMP_WAV.unlink()

    try:
        recording_process = subprocess.Popen(
            ["sox", "-d", "-r", "16000", "-c", "1", "-b", "16", str(TEMP_WAV)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        notify("VibeMic", "sox not found. Install: sudo apt install sox", "dialog-error")
        return

    is_recording = True
    update_tray("recording")
    notify("VibeMic", "Recording... Press PgDn to stop")


def stop_and_transcribe(tray, update_tray):
    """Stop recording, transcribe, optionally paraphrase, then type the result."""
    global recording_process, is_recording, config

    if not recording_process:
        is_recording = False
        update_tray("idle")
        return

    recording_process.send_signal(signal.SIGINT)
    try:
        recording_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        recording_process.kill()
        recording_process.wait()

    recording_process = None
    is_recording = False
    update_tray("transcribing")
    notify("VibeMic", "Transcribing...")

    if not TEMP_WAV.exists():
        notify("VibeMic", "No audio recorded. Check mic.", "dialog-error")
        update_tray("idle")
        return

    if TEMP_WAV.stat().st_size < MIN_FILE_SIZE:
        notify("VibeMic", "Too short, try again.", "dialog-warning")
        update_tray("idle")
        return

    config = load_config()
    readiness_issue = transcription_readiness_issue(config)
    if readiness_issue:
        notify("VibeMic", readiness_issue, "dialog-error")
        update_tray("idle")
        return

    try:
        provider = make_transcription_provider(config)
        text = provider.transcribe(TEMP_WAV, config).strip()
        if not text:
            notify("VibeMic", "No speech detected.", "dialog-warning")
            update_tray("idle")
            return

        original_text = text

        if config.get("paraphrase_enabled"):
            default_api_key = str(config.get("api_key", "")).strip()
            if not default_api_key:
                notify("VibeMic", "Paraphrase is enabled but Default API Key is missing.", "dialog-warning")
            else:
                update_tray("paraphrasing")
                notify("VibeMic", "Paraphrasing...")
                try:
                    para_prompt = config.get("paraphrase_prompt", DEFAULT_CONFIG["paraphrase_prompt"])
                    para_model = config.get("paraphrase_model", "gpt-4o-mini")
                    text = paraphrase_text(text, default_api_key, para_prompt, para_model)
                except Exception as exc:
                    notify("VibeMic", f"Paraphrase failed, using original: {str(exc)[:60]}", "dialog-warning")

        save_to_history(text, original=original_text)

        process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        process.communicate(text.encode("utf-8"))

        import time
        time.sleep(0.05)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], timeout=5)

        notify("VibeMic", f"Typed: {text[:60]}{'…' if len(text) > 60 else ''}")
        update_tray("idle")

    except Exception as exc:
        message = str(exc)
        if "401" in message or "Incorrect API key" in message or "authentication" in message.lower():
            notify("VibeMic", "Invalid transcription API key. Check Settings.", "dialog-error")
        elif "ECONNREFUSED" in message or "ENOTFOUND" in message or "Connection error" in message:
            notify("VibeMic", "Can't reach the transcription provider.", "dialog-error")
        else:
            notify("VibeMic", f"Error: {message[:120]}", "dialog-error")
        update_tray("idle")

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

    readiness_issue = transcription_readiness_issue(config)
    if readiness_issue:
        print(f"WARNING: {readiness_issue}")

    if config.get("paraphrase_enabled") and not str(config.get("api_key", "")).strip():
        print("WARNING: Paraphrase is enabled but Default API Key is missing.")

    if not any((Path(directory) / "sox").exists() for directory in os.environ.get("PATH", "").split(":")):
        print("ERROR: sox not found. Install: sudo apt install sox libsox-fmt-all")
        sys.exit(1)

    icons = {
        "idle": create_tray_icon((80, 140, 220, 255)),
        "recording": create_tray_icon((220, 40, 40, 255)),
        "transcribing": create_tray_icon((220, 160, 0, 255)),
        "paraphrasing": create_tray_icon((140, 80, 220, 255)),
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
            "paraphrasing": "VibeMic — Paraphrasing...",
        }
        tray.title = titles.get(state, titles["idle"])

    def open_history_click(icon, item):
        del icon, item
        open_history_dialog()

    def toggle_paraphrase(icon, item):
        del icon, item
        global config
        config["paraphrase_enabled"] = not config.get("paraphrase_enabled", False)
        save_config(config)
        state = "ON" if config["paraphrase_enabled"] else "OFF"
        notify("VibeMic", f"Paraphrase mode {state}")

    xdpy = None
    current_keycode = [None]

    def x11_grab(keycode):
        if not xdpy:
            return
        try:
            root_window = xdpy.screen().root
            for mod_mask in [0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask]:
                root_window.grab_key(keycode, mod_mask, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.flush()
        except Exception as exc:
            print(f"Warning: X11 grab failed: {exc}")

    def x11_ungrab(keycode):
        if not xdpy:
            return
        try:
            root_window = xdpy.screen().root
            for mod_mask in [0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask]:
                root_window.ungrab_key(keycode, mod_mask)
            xdpy.flush()
        except Exception:
            pass

    if xdisplay is not None and XK is not None:
        try:
            xdpy = xdisplay.Display()
            initial_keysym = KEY_NAME_TO_XK.get(config.get("hotkey", "page_down"), XK.XK_Next)
            current_keycode[0] = xdpy.keysym_to_keycode(initial_keysym)
            x11_grab(current_keycode[0])
            print(f"Hotkey '{config.get('hotkey', 'page_down')}' grabbed — key won't reach other apps.")
        except Exception as exc:
            print(f"Warning: Could not grab hotkey at X11 level: {exc}")
            print("Hotkey may still reach focused applications.")
    else:
        print("Warning: python-xlib not installed. Falling back to pynput-only hotkey handling.")

    def on_settings_save(new_config):
        global config
        config = normalize_config(new_config)

    def on_hotkey_change(new_key_name):
        global RECORD_KEY
        new_key = getattr(keyboard.Key, new_key_name, None)
        if new_key is None:
            return

        if current_keycode[0] is not None:
            x11_ungrab(current_keycode[0])

        new_keysym = KEY_NAME_TO_XK.get(new_key_name)
        if new_keysym and xdpy:
            new_keycode = xdpy.keysym_to_keycode(new_keysym)
            x11_grab(new_keycode)
            current_keycode[0] = new_keycode

        RECORD_KEY = new_key
        print(f"Hotkey changed to '{new_key_name}'")

    def open_settings_click(icon, item):
        del icon, item
        open_settings_dialog(on_settings_save, on_hotkey_change)

    def quit_app(icon, item):
        del item
        global recording_process
        if recording_process:
            recording_process.kill()
        if current_keycode[0] is not None:
            x11_ungrab(current_keycode[0])
        icon.stop()

    tray.menu = pystray.Menu(
        pystray.MenuItem("VibeMic", None, enabled=False),
        pystray.MenuItem("📋  History", open_history_click),
        pystray.MenuItem("⚙️  Settings...", open_settings_click),
        pystray.MenuItem("✍️  Paraphrase", toggle_paraphrase, checked=lambda item: config.get("paraphrase_enabled", False)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )

    def on_press(key):
        if key == RECORD_KEY:
            on_hotkey(tray, update_tray)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    provider = provider_spec_for_config(config)
    print("VibeMic Native running. Tray icon active.")
    print(f"Provider: {provider.display_name}")
    print(f"Model: {config.get('model')}")
    print(f"Language: {config.get('language') or 'auto'}")
    print(f"Paraphrase: {'ON' if config.get('paraphrase_enabled') else 'OFF'}")
    if provider.uses_remote_api:
        print(f"Transcription API key: {'set' if resolved_transcription_api_key(config) else 'missing'}")
    else:
        print(f"Local model: {resolved_local_whisper_model_path(config)}")
    tray.run()


if __name__ == "__main__":
    main()
