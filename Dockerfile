FROM python:3.8.15-bullseye

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 rsync -y

COPY utils/cluster/requirements.txt requirements.txt

RUN pip install torch torchvision torchaudio
RUN pip install -r requirements.txt
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
