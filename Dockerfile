FROM python:3.11-slim

WORKDIR /code 

COPY ./requirements.txt ./
RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN python -m venv venv
RUN chmod +x ./venv/bin/activate && ./venv/bin/activate
RUN pip install --no-cache-dir -r requirements.txt

RUN huggingface-cli login
RUN wget -O src/data/databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl

COPY ./src ./src

EXPOSE 8003

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]