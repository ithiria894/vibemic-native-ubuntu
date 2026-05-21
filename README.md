# VibeMic Native (Ubuntu)

[中文版](#中文) | English

![VibeMic](vibemic-slide.png)

System-wide voice-to-text for Ubuntu. Press `PgDn` to record, press again to transcribe, and VibeMic pastes the text into your current app.

This Ubuntu version now follows the same transcription-provider direction as the Mac app:

- `OpenAI`
- `Groq`
- `LiteLLM`
- `Custom OpenAI-compatible`
- `Local whisper.cpp`

It also includes a built-in **Model Library** in Settings, so each local model can be downloaded first and only then selected for use.

## Features

- **System-wide** — works in any X11 app, not just editors
- **One-key toggle** — `PgDn` starts and stops recording
- **Instant paste** — clipboard + `Ctrl+V`, no per-character typing delay
- **Transcript history** — browse, copy, and delete past transcripts from the tray menu
- **Native settings window** — configure provider, models, prompts, and hotkey from the tray icon
- **Provider abstraction** — switch between remote OpenAI-compatible APIs and local `whisper.cpp`
- **Local model downloads** — download presets directly from Settings before using them
- **Paraphrase mode** — optional rewrite pass after transcription using the default API key

## Local model presets

The built-in presets match the current Mac app:

- `Recommended - Large v3 Q5`
- `Fast - Large v3 Turbo Q5`
- `Fast Full - Large v3 Turbo`
- `Max - Large v3`
- `Cantonese Focus - Large v3 Cantonese Q8`
- `Cantonese Max - Large v3 Cantonese BF16`

For Cantonese mixed with lots of English, the default recommendation is still `Recommended - Large v3 Q5`. The Cantonese-specialized models can be better on pure Cantonese, but are usually less stable on English code-switch.

## Requirements

- Ubuntu 20.04+ (or Linux with X11)
- Python 3.8+
- `sox` for audio recording
- `xdotool` + `xclip` for clipboard paste
- `python3-xlib` for X11-level hotkey grab
- For remote providers: the relevant API key
- For local provider: a working `whisper-cli` binary from `whisper.cpp`

If `python3-xlib` is missing, VibeMic will still try to run with `pynput` hotkey fallback, but the key may also reach the focused app.

## Quick Start

```bash
git clone https://github.com/ithiria894/vibemic-native-ubuntu.git
cd vibemic-native-ubuntu
chmod +x setup.sh
./setup.sh
python3 vibemic.py
```

Then open **Settings** from the tray icon and choose one of these flows:

- `OpenAI` / `Groq` / `LiteLLM` / `Custom OpenAI-compatible`: add API key and model
- `Local whisper.cpp`: point to `whisper-cli`, download a model from **Model Library**, then save

## Manual Setup

```bash
sudo apt install sox libsox-fmt-all xdotool xclip libnotify-bin python3-tk python3-xlib
pip3 install --user openai pystray pynput Pillow
python3 vibemic.py
```

## Settings overview

| Setting | Description |
|---------|-------------|
| Default API Key | Fallback key for remote transcription providers and for paraphrase mode |
| Transcription Provider | `OpenAI`, `Groq`, `LiteLLM`, `Custom OpenAI-compatible`, or `Local whisper.cpp` |
| Remote Model | Suggested model list for the selected remote provider; still editable for custom endpoints |
| Transcription API Key | Optional provider-specific override |
| Base URL | Optional override for built-in providers, required for custom OpenAI-compatible endpoints |
| Local Binary Path | Path to `whisper-cli` |
| Local Model Preset | Slot to choose the active local preset |
| Local Model Path | Actual `.bin` file path; can be custom |
| Model Library | Download each preset before using it |
| Language | Auto-detect or pin to a specific code |
| Prompt | Hint text for likely languages or vocabulary |
| Temperature | Recognition temperature |
| Response Format | `json`, `text`, `srt`, `verbose_json`, `vtt` for remote providers |
| Paraphrase | Optional rewrite pass after transcription |

Settings are saved to `config.json`.

## Tray Menu

- **History** — browse and copy transcripts
- **Settings** — open the native settings window
- **Paraphrase** — toggle paraphrase mode quickly
- **Quit** — stop VibeMic

## How it pastes

Text is copied to the clipboard with `xclip` and pasted using `xdotool key ctrl+v`, so it stays fast even for long text and works well with CJK characters.

## Related

- [VibeMic Native macOS](https://github.com/agents-io/vibemic-native-macos)

## License

MIT

---

<a name="中文"></a>
## 中文

![VibeMic](vibemic-slide-zh.png)

Ubuntu 版全系統語音轉文字工具。按 `PgDn` 開始錄音，再按一次就轉錄並貼上到目前應用程式。

而家呢個 Ubuntu 版已經跟返 Mac 版嗰個方向：

- 支援多個 transcription provider
- 支援 `Local whisper.cpp`
- Settings 入面有 `Local Model Preset`
- 每個本機 model 都有下載位，下載完先用得

### 重點功能

- 全系統使用，不限 editor
- 一鍵錄音 / 停止
- 剪貼簿即時貼上
- Tray menu 內建 history
- Native settings UI
- Provider 可切換：`OpenAI`、`Groq`、`LiteLLM`、`Custom OpenAI-compatible`、`Local whisper.cpp`
- 本機 model library 直接下載

### 建議

如果你主要講 `廣東話 + 英文夾講`，建議先試：

- `Recommended - Large v3 Q5`

`Cantonese` 專門版可以留作比較，但通常冇咁穩定處理英文 code-switch。

### 安裝

```bash
git clone https://github.com/ithiria894/vibemic-native-ubuntu.git
cd vibemic-native-ubuntu
chmod +x setup.sh
./setup.sh
python3 vibemic.py
```

如果你想用本機模式：

1. 安裝好 `whisper.cpp` 嘅 `whisper-cli`
2. 喺 Settings 揀 `Local whisper.cpp`
3. 喺 `Model Library` 下載 model
4. 下載完成後先儲存使用

其他詳細設定請參考上面英文版。
