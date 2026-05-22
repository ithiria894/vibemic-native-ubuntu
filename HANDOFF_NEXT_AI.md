# VibeMic Ubuntu Handoff For Next AI

如果你係下一個接手 `vibemic-native-ubuntu` 嘅 AI，先睇呢份。

## 一句定位

Ubuntu 版已經唔再係純 OpenAI Whisper 小工具，而係跟返 Mac 版方向：

- transcription provider 已經抽象化
- settings 已經有 local / remote provider 切換
- 本機 model 已經有 preset 同 download 位
- 但真正喺 Ubuntu/X11 實機跑過未，尤其係 local `whisper.cpp`，仲未完全確認

所以而家最實際嘅態度係：

- `remote provider` 幾大機會即用得
- `local whisper.cpp` 值得試，但唔應該當必定 work

## 先睇邊幾份

1. `README.md`
2. `vibemic.py`
3. `setup.sh`
4. 呢份 `HANDOFF_NEXT_AI.md`

## 最新已完成狀態

最新基線 commit：

- `e7a8600` — `Add provider abstraction and local model downloads`

Ubuntu repo 而家已經有：

- `OpenAI`
- `Groq`
- `LiteLLM`
- `Custom OpenAI-compatible`
- `Local whisper.cpp`

local presets 亦已經加咗：

- `Recommended - Large v3 Q5`
- `Fast - Large v3 Turbo Q5`
- `Fast Full - Large v3 Turbo`
- `Max - Large v3`
- `Cantonese Focus - Large v3 Cantonese Q8`
- `Cantonese Max - Large v3 Cantonese BF16`

settings 入面每個 local model 都有：

- status
- download / use button
- 未下載唔畀 save 成 active local model

## 呢個 repo 最重要嘅現實

今次改動係喺 macOS 上完成，所以 verify 到嘅主要係：

- Python syntax
- import smoke test
- helper resolution
- config / provider / preset flow
- setup script syntax

未 verify 到嘅係：

- 真 Ubuntu tray behavior
- 真 X11 global hotkey grabbing
- `xdotool` / `xclip` paste path
- `whisper-cli` 喺 Ubuntu 上嘅實際 binary path
- local model download 完之後實際 transcription latency / stability

所以如果你喺 Ubuntu 機接手，唔好將「code 寫咗」當成「runtime 一定冇問題」。

## 接手時應該點做

### Path A：先求穩，先用 remote provider

如果目標係快啲用得，先唔好糾結 local：

1. 跑 `./setup.sh`
2. 開 `python3 vibemic.py`
3. Settings 揀：
   - `Groq`
   - 或 `OpenAI`
   - 或你自己已有嘅 `LiteLLM`
4. 填 key
5. 試 `PgDn` 錄音 / 停止 / paste

呢條路最 likely 先跑得通。

### Path B：試 local whisper.cpp

如果 Ubuntu 機資源夠、而且真係想零 API：

1. 先確認有 `whisper-cli`
2. Settings 揀 `Local whisper.cpp`
3. `Local Binary Path` 指去 `whisper-cli`
4. 喺 `Model Library` 下載：
   - 首選 `Recommended - Large v3 Q5`
5. Save 後試錄音

如果呢步失敗，最先懷疑：

- `whisper-cli` path 錯
- binary 唔可執行
- model 未真係落完
- Ubuntu 機上音訊裝置 / sox capture 有問題

## 對廣東話 + 英文 code-switch 嘅建議

預設仍然應該先試：

- `Recommended - Large v3 Q5`

原因：

- 對 `廣東話 + English` 夾講，通常比 Cantonese-specialized model 更穩
- `Cantonese` 專門版可以比較，但唔應該先當預設

## 關於 local model：唔好先入戲太深

呢個位係今次 handoff 最重要嘅取態。

如果 Ubuntu 機最後：

- 裝唔到 `whisper.cpp`
- 或 binary path 好麻煩
- 或速度太慢
- 或 X11 / audio stack 搞到好多 runtime 細 bug

咁就直接退返去：

- `Groq`
- `OpenAI`
- `LiteLLM`

即係話：

- architecture 已經畀你預備好
- local 係 bonus
- remote provider 唔係失敗，而係合理 fallback

## 重要 implementation 位

主要 logic 都喺：

- `vibemic.py`

關鍵區塊：

- provider metadata
- local model preset metadata
- config normalization / readiness checks
- `OpenAICompatibleTranscriptionProvider`
- `LocalWhisperCppTranscriptionProvider`
- settings dialog 入面嘅 provider switching
- settings dialog 入面嘅 `Model Library`

setup / onboarding 主要喺：

- `setup.sh`
- `README.md`

## 如果只做一件事

喺真 Ubuntu 機做一次完整 smoke：

1. launch app
2. open settings
3. save provider config
4. record
5. transcribe
6. paste into another app

先分清楚：

- 問題係 UI / hotkey / X11
- 定係 transcription provider
- 定係 local `whisper.cpp`

未做完呢一步之前，唔好太早改 architecture。
