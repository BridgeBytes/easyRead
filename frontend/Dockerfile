FROM python:3.12-slim

LABEL  "MAINTAINER" "Arnaud Kayonga <arnauldkayonga1@gmail.com>"
LABEL "Project" "EasyRead Together: An AI-powered document reader"

WORKDIR /frontend

EXPOSE 7860

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

COPY ./requirements.txt .

RUN pip install -r ./requirements.txt

CMD ["python", "main.py"]