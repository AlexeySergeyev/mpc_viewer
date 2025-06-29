# Running Flask Locally with Gunicorn and Nginx
⸻

✅ 1. Project Structure Example

myproject/
├── app.py
├── venv/
├── flaskapp.service      # (if testing systemd)
└── nginx/
    └── flaskapp.conf     # nginx config for local use


⸻

🧱 2. Create Your Flask App

app.py

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from Flask behind Gunicorn and Nginx!"


⸻

🐍 3. Setup Python Environment

python3 -m venv venv
source venv/bin/activate
pip install flask gunicorn

Test that it runs:

gunicorn --bind 127.0.0.1:8000 app:app

Open http://127.0.0.1:8000 → it should work.

⸻

🌐 4. Configure Nginx (as reverse proxy)

Create file:

sudo nano /etc/nginx/sites-available/flaskapp

Paste this:

server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:8000;
        include proxy_params;
        proxy_redirect off;
    }
}

Enable it:

sudo ln -s /etc/nginx/sites-available/flaskapp /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx

Go to http://localhost — Nginx will forward to Gunicorn → Flask.

⸻

🛠️ Optional: Run Gunicorn via systemd (mimic production)

Create file /etc/systemd/system/flaskapp.service:

[Unit]
Description=Gunicorn instance to serve Flask on localhost
After=network.target

[Service]
User=yourusername
WorkingDirectory=/home/yourusername/myproject
Environment="PATH=/home/yourusername/myproject/venv/bin"
ExecStart=/home/yourusername/myproject/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 app:app

[Install]
WantedBy=multi-user.target

Start it:

sudo systemctl daemon-reexec
sudo systemctl start flaskapp
sudo systemctl enable flaskapp


⸻

✅ Testing the Flow Locally

Component	URL	Purpose
Flask only	flask run or http://127.0.0.1:5000	dev only
Gunicorn	http://127.0.0.1:8000	WSGI server
Nginx	http://localhost	production-like access


⸻

🧪 Tips
	•	Logs:

sudo journalctl -u flaskapp.service
tail -f /var/log/nginx/access.log


	•	If something goes wrong, stop systemd and test Gunicorn manually:

gunicorn --bind 127.0.0.1:8000 app:app



⸻

Would you like a Docker-based local setup instead? I can provide a full example too.