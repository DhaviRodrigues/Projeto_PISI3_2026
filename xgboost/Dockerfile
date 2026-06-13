# Usa uma imagem oficial do Python levinha
FROM python:3.11-slim

# Define a pasta de trabalho dentro do servidor
WORKDIR /code

# Copia e instala os requisitos
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia todo o resto do seu código
COPY . .

# Expõe a porta 7860 (Padrão obrigatório do Hugging Face)
EXPOSE 7860

# Comando para rodar a aplicação usando o Gunicorn (lembre-se do caminho dashboard.dashboard)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "dashboard.dashboard:server"]