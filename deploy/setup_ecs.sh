#!/usr/bin/env bash
# =============================================================================
# Automated ECS Deployment Script for Alibaba Cloud (Ubuntu 22.04 / 24.04)
# Run on the ECS instance as root / sudo:
#   sudo bash deploy/setup_ecs.sh
# =============================================================================

set -e

echo "=== 1. Updating System Packages ==="
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    curl \
    git \
    build-essential

PROJECT_DIR="$(pwd)"

echo "=== 2. Creating Virtual Environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== 3. Configuring Nginx Reverse Proxy ==="
cp deploy/nginx.conf /etc/nginx/sites-available/etd.conf
ln -sf /etc/nginx/sites-available/etd.conf /etc/nginx/sites-enabled/etd.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

echo "=== 4. Setting Up Systemd Service for ETD ==="
cat <<EOF > /etc/systemd/system/etd.service
[Unit]
Description=Electricity Theft Detection Platform (ETD)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${PROJECT_DIR}/run.sh both
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable etd
systemctl restart etd

echo "============================================================"
echo " ETD Platform successfully deployed on Alibaba Cloud ECS!"
echo " Public Web Dashboard: http://$(curl -s ifconfig.me)"
echo " API Documentation:    http://$(curl -s ifconfig.me)/docs"
echo " Service Status:       systemctl status etd"
echo "============================================================"
