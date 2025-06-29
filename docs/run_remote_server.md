# Run Flask App with Nginx and Gunicorn

✅ Step-by-step to deploy Flask with Nginx and Gunicorn

1. Assumptions
	•	You already have:
	•	A Flask app (e.g., app.py)
	•	Domain name: alexeysergeyev.online pointed to 37.59.112.32
	•	A VPS with Ubuntu
	•	SSH access

⸻

🔧 2. Install Required Packages

sudo apt update
sudo apt install python3-pip python3-venv nginx


⸻

🐍 3. Set up your Flask App

Inside your app folder:

python3 -m venv venv
source venv/bin/activate
pip install flask gunicorn

Check your Flask app works locally:

python app.py
# or
gunicorn -w 4 -b 127.0.0.1:8000 app:app


⸻

🛠 4. Set up Gunicorn as a systemd service

Create a file:

sudo nano /etc/systemd/system/flaskapp.service

Paste this (replace youruser and yourpath):

[Unit]
Description=Gunicorn instance to serve Flask app
After=network.target

[Service]
User=youruser
Group=www-data
WorkingDirectory=/home/youruser/yourapp
Environment="PATH=/home/youruser/yourapp/venv/bin"
ExecStart=/home/youruser/yourapp/venv/bin/gunicorn --workers 3 --bind unix:flaskapp.sock -m 007 app:app

[Install]
WantedBy=multi-user.target

Enable and start:

sudo systemctl daemon-reexec
sudo systemctl start flaskapp
sudo systemctl enable flaskapp


⸻

🌐 5. Configure Nginx

Create a config:

sudo nano /etc/nginx/sites-available/flaskapp

Paste:

server {
    listen 80;
    server_name alexeysergeyev.online 37.59.112.32;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/youruser/yourapp/flaskapp.sock;
    }
}

Enable it:

sudo ln -s /etc/nginx/sites-available/flaskapp /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx


⸻

🔓 6. Allow Nginx through firewall

sudo ufw allow 'Nginx Full'
sudo ufw enable


⸻

🔁 7. Remove Default Page (optional)

sudo rm /etc/nginx/sites-enabled/default
sudo systemctl reload nginx


⸻

✅ Now test:
	•	Go to: http://alexeysergeyev.online
	•	Or IP: http://37.59.112.32

You should see your Flask app, not the Nginx welcome page.

⸻