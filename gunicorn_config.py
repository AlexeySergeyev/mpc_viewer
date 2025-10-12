# Gunicorn configuration file for MPC Viewer
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000

# Timeout - CRITICAL: Set to 10 minutes to handle long Miriade requests
# Miriade API can take several minutes to process large datasets
timeout = 600  # 10 minutes

keepalive = 5

# Logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

accesslog = os.path.join(log_dir, "gunicorn_access.log")
errorlog = os.path.join(log_dir, "gunicorn_error.log")
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "mpc_viewer"

# Server mechanics
daemon = False
pidfile = os.path.join(log_dir, "gunicorn.pid")
umask = 0

# Performance tuning
max_requests = 1000  # Restart workers after N requests (prevents memory leaks)
max_requests_jitter = 50  # Add randomness to max_requests

# Graceful timeout
graceful_timeout = 30

# SSL (uncomment and configure if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Debugging (set to True for development)
reload = False
reload_extra_files = []

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
