FROM python:3.10.14-slim-bullseye


WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg --fix-missing -y
RUN apt-get install libsm6 --fix-missing -y
RUN apt-get install libxext6 --fix-missing -y

# installing these dependencies first to make use of docker caching
RUN pip install tensorflow==2.9.1
RUN pip install Flask==2.0.2


RUN pip install opencv-python==4.6.0.66


COPY requirements.txt /app/requirements.txt

# RUN apt-get update
# RUN pip install --upgrade pip

RUN cd /app && pip install -r requirements.txt

# RUN pip install -r requirements.txt

COPY . .

# In final build we shall use the model by attaching model as a volume
# attach model to container as volume
# VOLUME /app/model

EXPOSE 5001


CMD ["python3", "app.py"]