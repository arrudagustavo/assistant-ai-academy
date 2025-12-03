# Usa uma imagem Python oficial leve
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /code

# Copia os arquivos de requisitos primeiro (para aproveitar cache)
COPY ./requirements.txt /code/requirements.txt

# Instala as dependências
# O --no-cache-dir deixa a imagem mais leve
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia o restante do código do projeto para dentro do container
COPY . /code

# Cria a pasta do banco de dados (para garantir permissão)
RUN mkdir -p /code/chroma_db && chmod 777 /code/chroma_db

# Libera a porta 8000 (onde o FastAPI roda)
EXPOSE 8000

# O comando para ligar o servidor quando estiver online
# Host 0.0.0.0 é fundamental para funcionar na nuvem
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]