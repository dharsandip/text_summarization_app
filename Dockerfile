FROM python:3.9.0
EXPOSE 8501
CMD mkdir -p /usr/app 
WORKDIR /usr/app
COPY . /usr/app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install tesseract-ocr
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["text-summarization_app.py"]