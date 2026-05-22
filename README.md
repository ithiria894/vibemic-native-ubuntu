# VibeMic Native (Ubuntu)

[中文版](#中文) | English

![VibeMic](vibemic-slide.png)

Free, open-source, system-wide voice-to-text for Ubuntu. Press `PgDn` to record, press again to transcribe, and VibeMic pastes the text into your current app. Works in any X11 application.

## Quick Start

```bash
git clone https://github.com/ithiria894/vibemic-native-ubuntu.git
cd vibemic-native-ubuntu
chmod +x setup.sh
./setup.sh
python3 vibemic.py
```

On first launch, VibeMic detects your hardware and walks you through setup. No manual configuration needed.

## Groq: Free Cloud Transcription

VibeMic works with [Groq](https://groq.com), which offers a **free** Whisper API with generous limits:

- ~8 hours of transcription per day, no credit card required
- Runs `whisper-large-v3-turbo` on Groq's LPU hardware, so results come back in under a second
- Sign up at [console.groq.com](https://console.groq.com) and create an API key at [console.groq.com/keys](https://console.groq.com/keys)

If you don't have a GPU, Groq is the recommended backend. VibeMic's first-launch wizard will guide you through it.

## Features

- **System-wide** -- works in any X11 app, not just editors
- **One-key toggle** -- `PgDn` starts and stops recording
- **Instant paste** -- clipboard + `Ctrl+V`, no per-character typing delay
- **First-launch wizard** -- auto-detects GPU and recommends the right backend
- **Multiple providers** -- OpenAI, Groq (free), LiteLLM, Custom OpenAI-compatible, Local whisper.cpp
- **Per-provider settings** -- each provider remembers its own API key and model
- **Transcript history** -- browse, copy, and delete past transcripts from the tray menu
- **Native settings** -- clean card-based UI, configure everything from the tray icon
- **Local model library** -- download Whisper models directly from Settings
- **Paraphrase mode** -- optional AI rewrite after transcription
- **Error logging** -- all errors logged to `~/.local/share/vibemic/vibemic.log`
- **Desktop notifications** -- native Ubuntu notifications for recording state

## Providers

| Provider | Speed | Offline | Cost | Setup |
|----------|-------|---------|------|-------|
| **Groq** | Fast (< 1s) | No | Free (8 hrs/day) | Sign up, paste key |
| **OpenAI** | Fast | No | $0.006/min | API key required |
| **LiteLLM** | Varies | No | Self-hosted | Local server URL |
| **Custom** | Varies | No | Varies | Base URL + key |
| **Local whisper.cpp** | Depends on GPU | Yes | Free | Install whisper-cli + download model |

## Local Model Presets

If you have a GPU (NVIDIA CUDA, AMD Vulkan, or Intel Vulkan), local transcription is fast and private:

| Preset | Size | Best for |
|--------|------|----------|
| Recommended - Large v3 Q5 | 1.1 GB | Best balance for multilingual speech |
| Fast - Large v3 Turbo Q5 | 548 MB | Fastest with strong quality |
| Fast Full - Large v3 Turbo | 1.5 GB | Turbo in full precision |
| Max - Large v3 | 3.1 GB | Highest quality |
| Cantonese Focus - Q8 | 1.6 GB | Fine-tuned for Cantonese |
| Cantonese Max - BF16 | 3.1 GB | Best Cantonese quality |

Without a GPU, local transcription on CPU is slow (~50 seconds per short clip). Use Groq instead.

## Requirements

- Ubuntu 20.04+ (or any Linux with X11)
- Python 3.8+
- `sox` for audio recording
- `xdotool` + `xclip` for clipboard paste
- `python3-xlib` for X11-level hotkey grab (optional, falls back to pynput)
- For Groq/OpenAI: an API key (Groq is free)
- For local: `whisper-cli` from [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

## Manual Setup

```bash
sudo apt install sox libsox-fmt-all xdotool xclip libnotify-bin python3-tk python3-xlib
pip3 install --user openai pystray pynput Pillow
python3 vibemic.py
```

## Settings

Right-click the tray icon and choose **Settings**. The UI is grouped into cards:

**Transcription** -- pick your provider, enter API key, select model. Each provider remembers its own key and model separately.

**Recognition** -- language hint, vocabulary hint for domain-specific words, response format (OpenAI/Custom only).

**Paraphrase** -- optional AI rewrite pass after transcription. Uses the same API key.

**Preferences** -- record hotkey (default: `PgDn`).

Settings are saved to `config.json`. Errors are logged to `~/.local/share/vibemic/vibemic.log`.

## How It Pastes

Text is copied to the clipboard with `xclip` and pasted using `xdotool key ctrl+v`. This is fast even for long text and works well with CJK characters.

## Related

- [VibeMic Native macOS](https://github.com/agents-io/vibemic-native-macos)

## License

MIT

---

<a name="中文"></a>
## 中文

![VibeMic](vibemic-slide-zh.png)

免費開源嘅 Ubuntu 全系統語音轉文字工具。按 `PgDn` 開始錄音，再按一次就轉錄並貼上到任何應用程式。

### 快速開始

```bash
git clone https://github.com/ithiria894/vibemic-native-ubuntu.git
cd vibemic-native-ubuntu
chmod +x setup.sh
./setup.sh
python3 vibemic.py
```

第一次啟動會自動偵測你嘅硬件，引導你完成設定。

### Groq：免費雲端轉錄

[Groq](https://groq.com) 提供免費嘅 Whisper API：

- 每日約 8 小時免費轉錄，唔使信用卡
- 用 `whisper-large-v3-turbo`，結果一秒內返回
- 去 [console.groq.com/keys](https://console.groq.com/keys) 申請免費 API key

冇 GPU 嘅話，Groq 係最推薦嘅選擇。

### 功能

- 全系統使用，任何 X11 app 都得
- 一鍵錄音 / 停止（`PgDn`）
- 剪貼簿即時貼上
- 首次啟動 wizard（自動偵測 GPU，推薦最適合嘅 backend）
- 多個 provider：OpenAI、Groq（免費）、LiteLLM、Custom、Local whisper.cpp
- 每個 provider 獨立記住 API key 同 model
- Tray menu 內建 history
- 原生 card-based 設定 UI
- 本機 model library 直接下載
- AI 改寫模式（Paraphrase）
- 錯誤自動記錄到 `~/.local/share/vibemic/vibemic.log`

### 建議

如果冇 GPU，用 **Groq**（免費、快）。

如果有 GPU，用 **Local whisper.cpp** + `Fast - Large v3 Turbo Q5`（離線、私密）。

如果主要講廣東話 + 英文夾講，model 揀 `Recommended - Large v3 Q5`。
