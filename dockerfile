FROM python:3.11-slim

WORKDIR /app

COPY requirements_cl.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

ENV CHAINLIT_PORT=8005

CMD ["chainlit", "run", "src/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]