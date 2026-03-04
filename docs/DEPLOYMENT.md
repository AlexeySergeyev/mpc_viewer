# MPC Viewer Deployment Guide (Docker + Nginx)

This guide is the canonical deployment flow for this project.

## 1. Architecture

- App container: Flask app served by Gunicorn on `127.0.0.1:5000` (published by Docker)
- Reverse proxy: Nginx on VPS (`80/443`) forwarding to `127.0.0.1:5000`
- TLS: Let's Encrypt certificates managed on VPS by Certbot

## 2. Important: Where Keys Are

Keys are not stored inside Docker image layers.

- SSH key for VPS access: local machine `~/.ssh/*`
- Git deploy key (if private repo): VPS user `~/.ssh/*`
- TLS private key: `/etc/letsencrypt/live/<domain>/privkey.pem` on VPS
- TLS full chain: `/etc/letsencrypt/live/<domain>/fullchain.pem` on VPS

If you "do not see your key in Docker deployment", this is expected.

## 3. First-Time VPS Setup

Run on VPS (Ubuntu):

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg nginx certbot python3-certbot-nginx

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
docker info
```

## 4. Clone and Run the App

Run on VPS:

```bash
cd /opt
sudo git clone https://github.com/AlexeySergeyev/mpc_viewer.git
sudo chown -R $USER:$USER /opt/mpc_viewer
cd /opt/mpc_viewer

docker compose -f compose.yml up --build -d
docker compose -f compose.yml ps
docker compose -f compose.yml logs --tail=100
curl -I http://127.0.0.1:5000
```

Expected: `HTTP/1.1 200 OK`.

## 5. Configure Nginx

Create `/etc/nginx/sites-available/mpc_viewer`:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name alexeysergeyev.online www.alexeysergeyev.online;
    return 301 https://alexeysergeyev.online$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name alexeysergeyev.online www.alexeysergeyev.online;

    ssl_certificate /etc/letsencrypt/live/alexeysergeyev.online/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/alexeysergeyev.online/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        send_timeout 600s;
    }
}
```

Enable and reload:

```bash
sudo ln -sf /etc/nginx/sites-available/mpc_viewer /etc/nginx/sites-enabled/alexeysergeyev.online
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 6. Issue/Reinstall Certificate

Run on VPS:

```bash
sudo certbot --nginx --redirect -d alexeysergeyev.online -d www.alexeysergeyev.online
```

If certbot asks:

- Use `1` to reinstall existing certificate
- Use `2` only when renewal is actually needed

## 7. Deploy Code Changes (Regular Workflow)

Local machine:

```bash
cd /Users/alexeysergeyev/code/python/mpc/projects/mpc_viewer
git add .
git commit -m "Describe change"
git push origin main
```

VPS:

```bash
cd /opt/mpc_viewer
git pull origin main
docker compose -f compose.yml up --build -d
docker compose -f compose.yml logs --tail=100
```

## 8. Conflict Recovery on VPS

If `git pull` fails with unresolved files and VPS has no important local edits:

```bash
cd /opt/mpc_viewer
git merge --abort || true
git fetch origin
git reset --hard origin/main
git clean -fd
docker compose -f compose.yml up --build -d
```

## 9. Health Checks

Run on VPS:

```bash
docker compose -f compose.yml ps
curl -I http://127.0.0.1:5000
sudo nginx -t
curl -I https://alexeysergeyev.online
```

## 10. Common Failures

- `permission denied /var/run/docker.sock`:
  - run `sudo usermod -aG docker $USER`, then relogin/newgrp
- `502 Bad Gateway`:
  - Nginx upstream is wrong or app container is down
  - verify `proxy_pass http://127.0.0.1:5000;`
- `invalid number of arguments in proxy_pass`:
  - syntax issue in Nginx site file
  - rewrite config and run `sudo nginx -t`
- DuckDB lock errors under Gunicorn:
  - use single worker process with threads (`workers = 1`, `gthread`)

