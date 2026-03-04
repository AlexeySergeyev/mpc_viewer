# Deploy Local Changes to VPS (Docker Compose)

This guide describes the exact workflow to deploy code changes from your local machine to your VPS.

## Assumptions

- Local repo path: `../mpc/projects/mpc_viewer`
- VPS repo path: `/opt/mpc_viewer`
- Branch: `main`
- App runs with Docker Compose (`compose.yml`)
- Nginx reverse proxy is already installed on VPS

## 1. Update Code Locally

Run on local machine:

```bash
cd ../mpc/projects/mpc_viewer

# optional check
git status

# stage and commit
git add .
git commit -m "Describe your change"

# push to remote
git push origin main
```

## 2. Pull and Deploy on VPS

SSH to VPS and run:

```bash
cd /opt/mpc_viewer
git pull origin main
docker compose -f compose.yml up --build -d
```

## 3. Verify Container and App

Run on VPS:

```bash
docker compose -f compose.yml ps
docker compose -f compose.yml logs --tail=120
curl -I http://127.0.0.1:5000
```

Expected:

- Container state is `Up`
- Local curl returns `HTTP/1.1 200 OK`

## 4. Verify Nginx and Public URL

Run on VPS:

```bash
sudo nginx -t
sudo systemctl reload nginx
curl -I https://alexeysergeyev.online
```

Expected:

- `nginx -t` is successful
- Public HTTPS returns `200` or valid redirect status

## 5. If `git pull` Fails with Merge Conflicts on VPS

If VPS repo is deployment-only and you do not need local VPS edits:

```bash
cd /opt/mpc_viewer
git merge --abort || true
git fetch origin
git reset --hard origin/main
git clean -fd
git pull origin main
docker compose -f compose.yml up --build -d
```

Warning: this deletes uncommitted VPS changes.

## 6. If Docker Permission Is Denied on VPS

If you see `permission denied ... /var/run/docker.sock`:

```bash
sudo usermod -aG docker $USER
newgrp docker
docker info
```

Then deploy again:

```bash
cd /opt/mpc_viewer
docker compose -f compose.yml up --build -d
```

## 7. If Domain Shows 502 / Connection Error

Run on VPS:

```bash
docker compose -f compose.yml ps
curl -I http://127.0.0.1:5000
sudo nginx -t
sudo systemctl status nginx --no-pager
sudo tail -n 80 /var/log/nginx/error.log
```

Most common cause:

- Nginx `proxy_pass` points to wrong upstream. It must proxy to:

`http://127.0.0.1:5000`

## 8. One-Command Deploy (Optional)

After changes are pushed, deploy quickly on VPS:

```bash
cd /opt/mpc_viewer && git pull origin main && docker compose -f compose.yml up --build -d
```

