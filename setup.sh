#!/bin/bash
# VibeMic Native — setup script for Ubuntu
set -e

echo "=== VibeMic Native Setup ==="

# System dependencies
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq sox libsox-fmt-all xdotool xclip libnotify-bin python3-tk python3-xlib

# Python dependencies
echo "Installing Python packages..."
pip3 install --user openai pystray pynput Pillow

# App directories
mkdir -p "$HOME/.local/share/vibemic/models" "$HOME/.cache/vibemic"

# Create .env if missing
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ""
    echo "⚠️  Created .env — edit it and add your OpenAI API key:"
    echo "    $SCRIPT_DIR/.env"
fi

# Make executable
chmod +x "$SCRIPT_DIR/vibemic.py"

# Desktop entry for autostart / app launcher
DESKTOP_FILE="$HOME/.local/share/applications/vibemic.desktop"
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=VibeMic
Comment=Voice-to-text — Press PgDn to record
Exec=python3 $SCRIPT_DIR/vibemic.py
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Audio;
StartupNotify=false
EOF

# Autostart entry (optional)
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cp "$DESKTOP_FILE" "$AUTOSTART_DIR/vibemic.desktop"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run:  python3 $SCRIPT_DIR/vibemic.py"
echo "To find: Search 'VibeMic' in your app launcher"
echo "Autostart: Enabled (runs on login)"
echo ""
echo "Usage: Press PgDn to start recording, PgDn again to stop & type."
echo ""
echo "Optional local transcription:"
echo "  1. Install whisper.cpp's whisper-cli somewhere on PATH"
echo "  2. Open Settings → Provider = Local whisper.cpp"
echo "  3. Download a model from the new Model Library section"
