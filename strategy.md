# VibeMic Strategy — Amphetamine of Voice Typing

## 定位

做全球第一個「裝完即用」嘅免費開源 system-wide voice typing app。目標係做到 Amphetamine 喺 keep-awake 嘅地位：人人都知、人人都用、完全免費。

## 點解有機會

- 30+ 個 app 做 voice-to-text，但冇贏家
- 冇一個 free app 做到「裝完即用」（唔使 API key、唔使 download model、唔使開帳號）
- Linux desktop voice typing 係明確嘅空白
- Wispr Flow 收 $15/mo，Superwhisper 收 $249 lifetime — wrapper 收呢個價，用戶怨氣大

## 核心原則

1. **完全免費** — 冇 pro version，冇 ads，冇 IAP
2. **零設置** — 裝完按 hotkey 即用，背後自動搞掂一切
3. **開源 MIT** — 任何人都可以 fork、改、用
4. **Privacy-first** — 能 local 就 local，要用 cloud 就透明講

## 技術方案

### Transcription Backend（智能 fallback）

```
App 啟動 → detect hardware
  ├─ 有 NVIDIA GPU → local faster-whisper (large-v3-turbo)
  ├─ 有 Apple Silicon → local whisper.cpp + Metal
  ├─ 有 AMD GPU + Vulkan → local whisper.cpp + Vulkan
  └─ 純 CPU（冇 GPU）→ Groq free tier（auto-provision）
```

用戶唔需要知道 backend 係咩。佢哋按 hotkey，講嘢，文字出現。

### Groq Free Tier 策略

- 每日 ~28,800 秒 audio（8 小時），20 requests/min
- 日常 voice typing 完全夠（平均每次錄音 5-15 秒）
- 風險：Groq 可以改 free tier → 要保留 local fallback
- App 第一次開嗰陣 auto-detect，如果 CPU-only 就 prompt 用戶「要唔要用免費 cloud？會快好多」或者「用 local（慢但離線）」

### Model 策略

| 情境 | Model | 大小 | 速度 |
|------|-------|------|------|
| GPU (NVIDIA/Apple Silicon) | large-v3-turbo-q5 | 548MB | 即時 |
| CPU fallback | base 或 small | 74-244MB | 堪用（~5-10 秒） |
| Cloud (Groq) | whisper-large-v3-turbo | server-side | 即時 |

第一次啟動時 auto-download model，唔係 bundle 入 installer（太大）。

## 平台優先順序

### Phase 1：Ubuntu（而家做）

原因：
- 已經有 working codebase
- Linux voice typing 係明確 gap，HN 新聞性最高
- 可以即時喺呢部機 test

要做：
- [ ] Auto-detect GPU 同選擇 backend
- [ ] 第一次啟動 wizard（選 local vs cloud，auto-download model）
- [ ] 打包成 `.deb`（apt install vibemic）
- [ ] 打包成 Flatpak（flathub 上架）
- [ ] 打包成 AppImage（download 即用）
- [ ] README 改成「Quick Start: 3 步即用」
- [ ] Floating pill overlay 已完成 ✅
- [ ] Desktop notifications 已完成 ✅
- [ ] Groq provider 已完成 ✅

### Phase 2：Mac（跟住做）

要做：
- [ ] Bundle local model 或 first-launch auto-download
- [ ] 去掉 API key 要求（local 做 default）
- [ ] Mac App Store 上架（sandbox、notarize）
- [ ] Homebrew cask `brew install --cask vibemic`
- [ ] 統一 Mac + Ubuntu 嘅 provider abstraction

### Phase 3：Distribution（兩個平台都 ready 之後）

- [ ] HN Show post：「Show HN: Free, open-source voice typing for Mac + Linux — no account, no API key」
- [ ] Dev.to article
- [ ] Reddit r/linux, r/macapps, r/opensource
- [ ] GitHub README 加 demo GIF
- [ ] Product Hunt launch

## 收入模式（Phase 4，如果有需要）

暫時唔收錢。如果用戶基數夠大：
- Donations（GitHub Sponsors / Buy Me a Coffee）
- VibeMic Cloud（$3/mo，畀唔想搞 local 嘅人）
- 企業版（team management、compliance、custom models）

呢啲全部係 optional，core app 永遠免費。

## 競爭定位

| | VibeMic | Wispr Flow | Superwhisper | VoiceInk |
|---|---|---|---|---|
| 價錢 | 免費 | $15/mo | $8.49/mo | $25 一次性 |
| 開源 | MIT | ❌ | ❌ | GPL3 |
| Linux | ✅ | ❌ | ❌ | ❌ |
| 離線 | ✅ | ❌ | ✅ | ✅ |
| 零設置 | ✅（目標） | 接近 | ❌ | 接近 |
| Cloud fallback | Groq free | OpenAI paid | BYOK | ❌ |

## 一句 pitch

**VibeMic：Free, open-source voice typing for Mac + Linux. No account. No API key. Just press the hotkey and talk.**
