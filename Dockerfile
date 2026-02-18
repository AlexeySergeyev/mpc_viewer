FROM python:3.12-slim

WORKDIR /asteroid-viewer

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
