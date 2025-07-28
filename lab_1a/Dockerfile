FROM continuumio/miniconda3
WORKDIR /app

COPY . .

RUN pip update & pip install numpy \
    matplotlib \
    torch