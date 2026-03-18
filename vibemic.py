#!/usr/bin/env python3
"""VibeMic Native — Voice-to-text for Ubuntu. Press PgDn to record, PgDn again to transcribe and type."""

import json
import os
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from openai import OpenAI
from pynput import keyboard
from PIL import Image, ImageDraw
from Xlib import X, display as xdisplay, XK

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

# ─── Config management ───
DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-4o-transcribe",
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
    S = 48
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mc = (255, 255, 255, 240)

    d.ellipse([0, 0, S - 1, S - 1], fill=color)
    d.rounded_rectangle([17, 7, 31, 27], radius=7, fill=mc)
    d.arc([11, 16, 37, 38], 0, 180, fill=mc, width=3)
    d.line([24, 37, 24, 42], fill=mc, width=3)
    d.line([18, 42, 30, 42], fill=mc, width=3)

    return img


# ─── Theme constants ───
BG = "#1a1a2e"
FG = "#e0e0e0"
ACCENT = "#64b5f6"
INPUT_BG = "#16213e"
BORDER = "#2a2a4a"


def _apply_theme(root):
    """Apply dark theme to a tkinter root/toplevel."""
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TLabel", background=BG, foreground=FG, font=("sans-serif", 10))
    style.configure("TButton", background=INPUT_BG, foreground=FG,
                    bordercolor=BORDER, relief="flat", padding=(10, 6))
    style.map("TButton", background=[("active", "#252550")])
    style.configure("Accent.TButton", background=ACCENT, foreground=BG,
                    font=("sans-serif", 10, "bold"))
    style.map("Accent.TButton", background=[("active", "#90caf9")])
    style.configure("TEntry", fieldbackground=INPUT_BG, foreground=FG,
                    bordercolor=BORDER, insertcolor=FG)
    style.configure("TCombobox", fieldbackground=INPUT_BG, foreground=FG,
                    background=INPUT_BG, selectbackground=INPUT_BG, selectforeground=ACCENT)
    style.map("TCombobox",
              fieldbackground=[("readonly", INPUT_BG)],
              foreground=[("readonly", FG)],
              selectbackground=[("readonly", INPUT_BG)])
    style.configure("TScale", background=BG, troughcolor=INPUT_BG, slidercolor=ACCENT)
    style.configure("TCheckbutton", background=BG, foreground=FG, focuscolor="",
                    indicatorcolor=INPUT_BG)
    style.map("TCheckbutton", background=[("active", BG)],
              indicatorcolor=[("selected", ACCENT)])
    style.configure("TScrollbar", background=INPUT_BG, troughcolor=BG,
                    bordercolor=BG, arrowcolor=FG)


def _label(parent, text, size=10, bold=False, color=None):
    font = ("sans-serif", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, bg=BG, fg=color or FG, font=font)


def _text_widget(parent, height=2):
    return tk.Text(
        parent, height=height, bg=INPUT_BG, fg=FG,
        insertbackground=FG, relief="flat", bd=1,
        font=("sans-serif", 10), padx=8, pady=6,
        wrap="word", highlightbackground=BORDER, highlightthickness=1,
    )


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

        # ── Scrollable canvas wrapper ──
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, width=500)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=BG, padx=24, pady=20)
        win_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_frame_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── Title ──
        _label(frame, "VibeMic Settings", size=16, bold=True, color=ACCENT).pack(anchor="w", pady=(0, 16))

        def section(text):
            _label(frame, text, size=10, bold=True).pack(anchor="w", pady=(12, 2))

        def hint(text):
            tk.Label(frame, text=text, bg=BG, fg="#888888",
                     font=("sans-serif", 9)).pack(anchor="w")

        def divider():
            tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(16, 4))

        # ── API Key ──
        section("OpenAI API Key")
        key_frame = tk.Frame(frame, bg=BG)
        key_frame.pack(fill="x")
        api_var = tk.StringVar(value=cfg.get("api_key", ""))
        api_entry = ttk.Entry(key_frame, textvariable=api_var, show="•", width=44)
        api_entry.pack(side="left", fill="x", expand=True)

        def toggle_key():
            api_entry.config(show="" if api_entry.cget("show") else "•")

        ttk.Button(key_frame, text="Show", command=toggle_key).pack(side="left", padx=(6, 0))

        # ── Transcription Model ──
        section("Transcription Model")
        model_var = tk.StringVar(value=cfg.get("model", "gpt-4o-transcribe"))
        ttk.Combobox(frame, textvariable=model_var, values=WHISPER_MODELS,
                     state="readonly", width=42).pack(fill="x")

        # ── Language ──
        section("Language")
        lang_names = [name for name, _ in LANGUAGES]
        lang_codes = [code for _, code in LANGUAGES]
        cur_code = cfg.get("language", "")
        cur_idx = lang_codes.index(cur_code) if cur_code in lang_codes else 0
        lang_var = tk.StringVar(value=lang_names[cur_idx])
        ttk.Combobox(frame, textvariable=lang_var, values=lang_names,
                     state="readonly", width=42).pack(fill="x")

        # ── Transcription Prompt ──
        section("Transcription Prompt")
        hint("Hint for Whisper — expected languages or vocabulary")
        prompt_text = _text_widget(frame, height=2)
        prompt_text.insert("1.0", cfg.get("prompt", ""))
        prompt_text.pack(fill="x", pady=(2, 0))

        # ── Temperature ──
        section("Temperature")
        temp_frame = tk.Frame(frame, bg=BG)
        temp_frame.pack(fill="x")
        temp_var = tk.DoubleVar(value=cfg.get("temperature", 0))
        temp_val_label = tk.Label(temp_frame, text=f"{temp_var.get():.1f}",
                                   bg=BG, fg=ACCENT, font=("sans-serif", 10, "bold"), width=4)
        temp_val_label.pack(side="right")

        def on_temp(v):
            temp_val_label.config(text=f"{float(v):.1f}")

        ttk.Scale(temp_frame, from_=0, to=1, variable=temp_var,
                  command=on_temp, orient="horizontal").pack(side="left", fill="x", expand=True)

        # ── Response Format ──
        section("Response Format")
        fmt_var = tk.StringVar(value=cfg.get("response_format", "json"))
        ttk.Combobox(frame, textvariable=fmt_var, values=RESPONSE_FORMATS,
                     state="readonly", width=42).pack(fill="x")

        # ── Hotkey ──
        section("Record Hotkey")
        hint("Click 'Change', then press any special key (PgDn, F-key, Home, etc.)")
        hotkey_frame = tk.Frame(frame, bg=BG)
        hotkey_frame.pack(fill="x", pady=(2, 0))
        hotkey_var = tk.StringVar(value=cfg.get("hotkey", "page_down"))
        hotkey_display = ttk.Entry(hotkey_frame, textvariable=hotkey_var,
                                   state="readonly", width=20)
        hotkey_display.pack(side="left")
        capturing = [False]
        capture_listener = [None]

        def start_capture():
            if capturing[0]:
                return
            capturing[0] = True
            change_btn.config(text="Press a key...", state="disabled")

            def on_key(key):
                if not capturing[0]:
                    return False
                try:
                    key_name = key.name  # special key e.g. "page_down", "f9"
                except AttributeError:
                    key_name = None
                if key_name and key_name in KEY_NAME_TO_XK:
                    hotkey_var.set(key_name)
                    root.after(0, lambda: change_btn.config(text="Change", state="normal"))
                    capturing[0] = False
                    return False
                # ignore non-special or unsupported keys, keep listening
                return None

            kl = keyboard.Listener(on_press=on_key)
            kl.daemon = True
            kl.start()
            capture_listener[0] = kl

        change_btn = ttk.Button(hotkey_frame, text="Change", command=start_capture)
        change_btn.pack(side="left", padx=(8, 0))

        # ══ Paraphrase Section ══
        divider()
        _label(frame, "Paraphrase", size=13, bold=True, color=ACCENT).pack(anchor="w", pady=(4, 0))
        hint("After transcription, rewrite text with an AI prompt before typing")

        para_enabled_var = tk.BooleanVar(value=cfg.get("paraphrase_enabled", False))
        ttk.Checkbutton(frame, text="Enable paraphrase mode",
                        variable=para_enabled_var).pack(anchor="w", pady=(8, 0))

        section("Paraphrase Prompt")
        hint("Instructions for how to rewrite the transcript")
        para_prompt_text = _text_widget(frame, height=4)
        para_prompt_text.insert("1.0", cfg.get("paraphrase_prompt", DEFAULT_CONFIG["paraphrase_prompt"]))
        para_prompt_text.pack(fill="x", pady=(2, 0))

        section("Paraphrase Model")
        para_model_var = tk.StringVar(value=cfg.get("paraphrase_model", "gpt-4o-mini"))
        ttk.Combobox(frame, textvariable=para_model_var, values=CHAT_MODELS,
                     state="readonly", width=42).pack(fill="x")

        # ── Buttons ──
        tk.Frame(frame, bg=BG, height=12).pack()
        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(fill="x", pady=(0, 4))

        def do_save():
            api_key = api_var.get().strip()
            if not api_key:
                messagebox.showerror("VibeMic", "API Key is required.", parent=root)
                return

            lang_name = lang_var.get()
            lang_code = next((c for n, c in LANGUAGES if n == lang_name), "")

            new_cfg = {
                "api_key": api_key,
                "model": model_var.get(),
                "language": lang_code,
                "prompt": prompt_text.get("1.0", "end-1c").strip(),
                "temperature": round(temp_var.get(), 1),
                "response_format": fmt_var.get(),
                "hotkey": hotkey_var.get(),
                "paraphrase_enabled": para_enabled_var.get(),
                "paraphrase_prompt": para_prompt_text.get("1.0", "end-1c").strip(),
                "paraphrase_model": para_model_var.get(),
            }
            save_config(new_cfg)
            if on_save:
                on_save(new_cfg)
            if on_hotkey_change:
                on_hotkey_change(new_cfg["hotkey"])
            notify("VibeMic", "Settings saved!")
            root.destroy()

        ttk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btn_frame, text="Save", command=do_save, style="Accent.TButton").pack(side="right")

        # Set a reasonable max height
        root.update_idletasks()
        screen_h = root.winfo_screenheight()
        win_h = min(frame.winfo_reqheight() + 40, int(screen_h * 0.85))
        root.geometry(f"520x{win_h}")

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

        # Header
        header = tk.Frame(root, bg=BG, padx=16, pady=12)
        header.pack(fill="x")
        _label(header, "VibeMic History", size=15, bold=True, color=ACCENT).pack(side="left")

        count_label = tk.Label(header, text="", bg=BG, fg="#888888",
                                font=("sans-serif", 10))
        count_label.pack(side="left", padx=(12, 0))

        # Scrollable entries
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

        def on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        card_widgets = []

        def refresh():
            for w in card_widgets:
                w.destroy()
            card_widgets.clear()
            history = load_history()
            count_label.config(text=f"{len(history)} transcript{'s' if len(history) != 1 else ''}")
            if not history:
                lbl = _label(inner, "No transcripts yet. Press PgDn to record!", color="#888888")
                lbl.pack(pady=40)
                card_widgets.append(lbl)
                return
            for i, entry in enumerate(history):
                text = entry.get("text", "")
                original = entry.get("original")
                ts = entry.get("timestamp", "")
                build_card(i, text, ts, original)

        def build_card(i, text, ts, original=None):
            card_bg = "#1e2d45" if original else INPUT_BG
            card = tk.Frame(inner, bg=card_bg, padx=12, pady=10,
                             highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", pady=(0, 6))
            card_widgets.append(card)

            row = tk.Frame(card, bg=card_bg)
            row.pack(fill="x")

            # Show paraphrase badge if this entry was paraphrased
            ts_text = f"✍️ {ts}" if original else ts
            tk.Label(row, text=ts_text, bg=card_bg, fg="#888888",
                      font=("sans-serif", 9)).pack(side="left")

            def make_delete(idx):
                def do():
                    delete_history_entry(idx)
                    refresh()
                return do

            def make_copy(t):
                def do():
                    root.clipboard_clear()
                    root.clipboard_append(t)
                    root.after(3000, root.clipboard_clear)
                return do

            ttk.Button(row, text="Delete", command=make_delete(i)).pack(side="right", padx=(4, 0))
            ttk.Button(row, text="Copy", command=make_copy(text)).pack(side="right")

            # Paraphrased text (main)
            tk.Label(card, text=text, bg=card_bg, fg=FG,
                      font=("sans-serif", 11), justify="left",
                      wraplength=560, anchor="w").pack(fill="x", pady=(6, 0), anchor="w")

            # Original transcript (if paraphrased)
            if original:
                tk.Label(card, text="Original:", bg=card_bg, fg="#888888",
                          font=("sans-serif", 9, "bold")).pack(anchor="w", pady=(8, 0))
                tk.Label(card, text=original, bg=card_bg, fg="#aaaaaa",
                          font=("sans-serif", 10), justify="left",
                          wraplength=560, anchor="w").pack(fill="x", anchor="w")

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
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        notify("VibeMic", "sox not found. Install: sudo apt install sox", "dialog-error")
        return

    is_recording = True
    update_tray("recording")
    notify("VibeMic", "Recording... Press PgDn to stop")


def stop_and_transcribe(tray, update_tray):
    """Stop recording, send to Whisper, optionally paraphrase, type the result."""
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
    notify("VibeMic", "Transcribing...")

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

        original_text = text

        # ── Paraphrase ──
        if config.get("paraphrase_enabled"):
            update_tray("paraphrasing")
            notify("VibeMic", "Paraphrasing...")
            try:
                para_prompt = config.get("paraphrase_prompt", DEFAULT_CONFIG["paraphrase_prompt"])
                para_model = config.get("paraphrase_model", "gpt-4o-mini")
                text = paraphrase_text(text, api_key, para_prompt, para_model)
            except Exception as e:
                notify("VibeMic", f"Paraphrase failed, using original: {str(e)[:60]}", "dialog-warning")

        # Save to history (include original transcript if paraphrased)
        save_to_history(text, original=original_text)

        # Paste via clipboard
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE
        )
        proc.communicate(text.encode("utf-8"))
        import time
        time.sleep(0.05)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], timeout=5)
        notify("VibeMic", f"Typed: {text[:60]}{'…' if len(text) > 60 else ''}")
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

    if not any((Path(d) / "sox").exists() for d in os.environ.get("PATH", "").split(":")):
        print("ERROR: sox not found. Install: sudo apt install sox libsox-fmt-all")
        sys.exit(1)

    icons = {
        "idle": create_tray_icon((80, 140, 220, 255)),          # Blue
        "recording": create_tray_icon((220, 40, 40, 255)),      # Red
        "transcribing": create_tray_icon((220, 160, 0, 255)),   # Orange
        "paraphrasing": create_tray_icon((140, 80, 220, 255)),  # Purple
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
        open_history_dialog()

    def toggle_paraphrase(icon, item):
        global config
        config["paraphrase_enabled"] = not config.get("paraphrase_enabled", False)
        save_config(config)
        state = "ON" if config["paraphrase_enabled"] else "OFF"
        notify("VibeMic", f"Paraphrase mode {state}")

    # Grab hotkey at X11 level
    xdpy = None
    current_keycode = [None]  # mutable so regrab can update it

    def x11_grab(keycode):
        if not xdpy:
            return
        try:
            root_win = xdpy.screen().root
            for mod_mask in [0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask]:
                root_win.grab_key(keycode, mod_mask, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.flush()
        except Exception as e:
            print(f"Warning: X11 grab failed: {e}")

    def x11_ungrab(keycode):
        if not xdpy:
            return
        try:
            root_win = xdpy.screen().root
            for mod_mask in [0, X.Mod2Mask, X.LockMask, X.Mod2Mask | X.LockMask]:
                root_win.ungrab_key(keycode, mod_mask)
            xdpy.flush()
        except Exception:
            pass

    try:
        xdpy = xdisplay.Display()
        initial_xk = KEY_NAME_TO_XK.get(config.get("hotkey", "page_down"), XK.XK_Next)
        current_keycode[0] = xdpy.keysym_to_keycode(initial_xk)
        x11_grab(current_keycode[0])
        print(f"Hotkey '{config.get('hotkey', 'page_down')}' grabbed — key won't reach other apps.")
    except Exception as e:
        print(f"Warning: Could not grab hotkey at X11 level: {e}")
        print("Hotkey may still reach focused applications.")

    def on_settings_save(new_config):
        global config
        config = new_config

    def on_hotkey_change(new_key_name):
        global RECORD_KEY
        new_key = getattr(keyboard.Key, new_key_name, None)
        if new_key is None:
            return
        # Ungrab old, grab new at X11 level
        if current_keycode[0] is not None:
            x11_ungrab(current_keycode[0])
        new_xk = KEY_NAME_TO_XK.get(new_key_name)
        if new_xk and xdpy:
            new_kc = xdpy.keysym_to_keycode(new_xk)
            x11_grab(new_kc)
            current_keycode[0] = new_kc
        # Update pynput listener key
        RECORD_KEY = new_key
        print(f"Hotkey changed to '{new_key_name}'")

    def open_settings_click(icon, item):
        open_settings_dialog(on_settings_save, on_hotkey_change)

    def quit_app(icon, item):
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
        pystray.MenuItem(
            "✍️  Paraphrase",
            toggle_paraphrase,
            checked=lambda item: config.get("paraphrase_enabled", False),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )

    def on_press(key):
        if key == RECORD_KEY:
            on_hotkey(tray, update_tray)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    print("VibeMic Native running. Tray icon active.")
    print(f"Config: model={config.get('model')}, language={config.get('language') or 'auto'}")
    print(f"API key: {'set' if config.get('api_key') else 'missing — open Settings'}")
    print(f"Paraphrase: {'ON' if config.get('paraphrase_enabled') else 'OFF'}")
    tray.run()


if __name__ == "__main__":
    main()
