FROM python:3.10.9-bullseye

WORKDIR /usr/src/app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8080
