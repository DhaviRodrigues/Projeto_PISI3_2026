FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/dashboard

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "--chdir", "/home/user/app/dashboard", "dashboard:server"]