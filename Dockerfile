FROM python:3.9.1-slim
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN pip install -r requirements.txt
