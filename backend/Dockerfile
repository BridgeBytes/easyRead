FROM nvcr.io/nvidia/pytorch:25.11-py3

LABEL  "MAINTAINER" "Arnaud Kayonga <arnauldkayonga1@gmail.com>"
LABEL "Project" "EasyRead Together: An AI-powered document reader"

WORKDIR /backend

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

CMD ["uv", "run", "uvicorn","api:api","--port","8000","--host","0.0.0.0", "--reload"]

