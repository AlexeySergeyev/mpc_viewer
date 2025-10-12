# MPC Viewer - Production Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install gunicorn  # Production server
```

### 2. Run with Gunicorn (Recommended for Production)

```bash
# Using the provided configuration file (recommended)
gunicorn -c gunicorn_config.py app:app

# Or with command-line options
gunicorn --timeout 600 --workers 4 --bind 0.0.0.0:5000 app:app
```

### 3. Access the Application

Open your browser and navigate to:
- Local: `http://localhost:5000`
- Network: `http://your-server-ip:5000`

## Configuration Files

### gunicorn_config.py

The provided configuration file includes:
- **Timeout**: 600 seconds (10 minutes) - Critical for Miriade requests
- **Workers**: Automatically calculated based on CPU cores
- **Logging**: Saves to `./logs/` directory
- **Performance**: Auto-restart workers to prevent memory leaks

### Environment Variables (Optional)

Create a `.env` file for custom configuration:

```bash
# Flask configuration
FLASK_ENV=production
FLASK_DEBUG=0

# Server configuration
HOST=0.0.0.0
PORT=5000

# Logging level
LOG_LEVEL=INFO
```

## Systemd Service (Linux)

For automatic startup on server boot:

### 1. Create Service File

Create `/etc/systemd/system/mpc_viewer.service`:

```ini
[Unit]
Description=MPC Viewer Application
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/code/python/mpc_viewer
Environment="PATH=/home/ubuntu/code/python/venvs/mpc/bin"
ExecStart=/home/ubuntu/code/python/venvs/mpc/bin/gunicorn -c gunicorn_config.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable mpc_viewer

# Start service
sudo systemctl start mpc_viewer

# Check status
sudo systemctl status mpc_viewer

# View logs
sudo journalctl -u mpc_viewer -f
```

### 3. Service Management

```bash
# Stop service
sudo systemctl stop mpc_viewer

# Restart service
sudo systemctl restart mpc_viewer

# Reload configuration (graceful)
sudo systemctl reload mpc_viewer

# Disable service
sudo systemctl disable mpc_viewer
```

## Nginx Reverse Proxy (Optional but Recommended)

### 1. Install Nginx

```bash
sudo apt update
sudo apt install nginx
```

### 2. Configure Nginx

Create `/etc/nginx/sites-available/mpc_viewer`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Change this

    # Increase timeouts for long-running Miriade requests
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
    send_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed in future)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files (optional optimization)
    location /static {
        alias /home/ubuntu/code/python/mpc_viewer/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Logs
    access_log /var/log/nginx/mpc_viewer_access.log;
    error_log /var/log/nginx/mpc_viewer_error.log;
}
```

### 3. Enable Site

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/mpc_viewer /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

### 4. SSL/HTTPS with Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (certbot sets this up automatically)
sudo certbot renew --dry-run
```

## Firewall Configuration

```bash
# Allow HTTP
sudo ufw allow 80/tcp

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow SSH (if not already allowed)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

## Monitoring

### Check Application Status

```bash
# Check if gunicorn is running
ps aux | grep gunicorn

# Check listening ports
sudo netstat -tlnp | grep 5000

# Check logs
tail -f logs/gunicorn_error.log
tail -f logs/gunicorn_access.log
tail -f logs/$(date +%Y-%m-%d).log  # Application logs
```

### Log Rotation

Create `/etc/logrotate.d/mpc_viewer`:

```
/home/ubuntu/code/python/mpc_viewer/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    missingok
    sharedscripts
    postrotate
        systemctl reload mpc_viewer > /dev/null 2>&1 || true
    endscript
}
```

## Troubleshooting

### Worker Timeout Errors

**Symptom:**
```
[CRITICAL] WORKER TIMEOUT (pid:XXXXX)
```

**Solution:**
- Increase timeout in `gunicorn_config.py` (already set to 600s)
- Check if Miriade API is responding slowly
- Review logs for specific errors

### Database Locked Errors

**Symptom:**
```
database is locked
```

**Solution:**
- Reduce number of workers (DuckDB has limited concurrency)
- Use `workers = 2` in gunicorn_config.py for DuckDB

### Memory Issues

**Symptom:**
- Slow performance
- Workers killed by OOM killer

**Solution:**
- Reduce `max_requests` in gunicorn_config.py
- Reduce number of workers
- Increase server RAM

### Port Already in Use

**Symptom:**
```
[ERROR] Address already in use
```

**Solution:**
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill the process
kill -9 PID

# Or change port in gunicorn_config.py
bind = "0.0.0.0:8000"
```

## Performance Optimization

### 1. Database Location

For better performance, ensure DuckDB files are on fast storage:
- SSD preferred over HDD
- Local storage preferred over network storage

### 2. Worker Count

Adjust based on your server:
```python
# In gunicorn_config.py
# For CPU-intensive tasks
workers = multiprocessing.cpu_count() * 2 + 1

# For I/O-intensive tasks (database operations)
workers = min(4, multiprocessing.cpu_count())
```

### 3. Caching

The application already caches data in DuckDB:
- First fetch: Slow (fetches from API)
- Subsequent fetches: Fast (loads from database)

## Security Considerations

### 1. Run as Non-Root User

Never run as root. Create dedicated user:
```bash
sudo useradd -r -s /bin/false mpc_viewer
```

### 2. Limit File Permissions

```bash
# Application directory
chmod 755 /home/ubuntu/code/python/mpc_viewer

# Database directory
chmod 755 db/
chmod 644 db/*.duckdb

# Logs directory
chmod 755 logs/
chmod 644 logs/*.log
```

### 3. Environment Variables

Store sensitive data in environment variables, not in code:
```bash
export SECRET_KEY='your-secret-key'
export DATABASE_URL='sqlite:///db/metadata.duckdb'
```

## Backup Strategy

### 1. Database Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/mpc_viewer"
DB_DIR="/home/ubuntu/code/python/mpc_viewer/db"

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/mpc_viewer_db_$DATE.tar.gz $DB_DIR
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

Add to crontab:
```bash
crontab -e
# Add:
0 2 * * * /path/to/backup_script.sh
```

### 2. Application Backup

```bash
# Backup application code (excluding venv and logs)
tar --exclude='venv' --exclude='logs' --exclude='__pycache__' \
    -czf mpc_viewer_app_$(date +%Y%m%d).tar.gz \
    /home/ubuntu/code/python/mpc_viewer
```

## Update Procedure

```bash
# 1. Pull latest changes
cd /home/ubuntu/code/python/mpc_viewer
git pull origin main

# 2. Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# 3. Restart service
sudo systemctl restart mpc_viewer

# 4. Check status
sudo systemctl status mpc_viewer
```

## Summary

**Minimal Production Setup:**
```bash
# 1. Install and configure
pip install -r requirements.txt
pip install gunicorn

# 2. Run
gunicorn -c gunicorn_config.py app:app
```

**Recommended Production Setup:**
1. ✅ Gunicorn with config file
2. ✅ Systemd service
3. ✅ Nginx reverse proxy
4. ✅ SSL/HTTPS
5. ✅ Firewall configuration
6. ✅ Log rotation
7. ✅ Automated backups

For more details, see:
- `docs/gunicorn_timeout_fix.md` - Timeout configuration
- `docs/run_remote_server.md` - Remote deployment guide
- `gunicorn_config.py` - Production configuration
