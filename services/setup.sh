#!/bin/bash
# Golf PrizePicks Scraper — systemd setup
# Run: sudo bash services/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$SCRIPT_DIR/golf-prizepicks-scraper.service"

echo "=== Golf PrizePicks Scraper Setup ==="
echo "Project: $PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv "$PROJECT_DIR/venv" 2>/dev/null || python3 -m venv "$PROJECT_DIR/venv"
fi

# Install dependencies
echo "Installing dependencies..."
"$PROJECT_DIR/venv/bin/pip" install -q --upgrade pip
"$PROJECT_DIR/venv/bin/pip" install -q requests sqlalchemy python-dotenv rich

# Create log and data directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data/cache"

# Install systemd service
echo "Installing systemd service..."
cp "$SERVICE_FILE" /etc/systemd/system/golf-prizepicks-scraper.service
systemctl daemon-reload
systemctl enable golf-prizepicks-scraper
systemctl start golf-prizepicks-scraper

echo ""
echo "=== Setup complete ==="
echo "Service status:"
systemctl status golf-prizepicks-scraper --no-pager -l || true
echo ""
echo "Useful commands:"
echo "  systemctl status golf-prizepicks-scraper"
echo "  journalctl -u golf-prizepicks-scraper -f"
echo "  tail -f $PROJECT_DIR/logs/prizepicks_scraper.log"
