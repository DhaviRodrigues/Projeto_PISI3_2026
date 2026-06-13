FROM python:3.11-slim

# Dependência do sistema para LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root (exigido pelo Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app:/home/user/app/dashboard

WORKDIR /home/user/app

# Copia e instala dependências
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia o restante do código
COPY --chown=user . .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "dashboard.dashboard:server"]