FROM --platform=linux/amd64 python:3.13.5-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5000

CMD ["sh", "-c", "PYTHONPATH=. python src/deploy/backend.py"]