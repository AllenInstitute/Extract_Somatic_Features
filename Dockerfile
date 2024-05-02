FROM continuumio/miniconda3:4.8.3
SHELL ["/bin/bash", "-c"]
RUN mkdir /app
WORKDIR /app
RUN apt-get update --allow-releaseinfo-change
RUN apt-get -y install build-essential libcgal-dev libassimp-dev libgl1-mesa-dev mesa-utils libgl1-mesa-glx
RUN conda config --add channels conda-forge && conda config --set channel_priority strict
COPY requirements.txt /app
RUN pip install -U numpy
RUN pip install -r requirements.txt
RUN pip install task-queue
COPY task_worker.py /app
COPY test_cv.py /app
COPY test_input.json /app
RUN mkdir /app/extract_somatic_features
COPY ./extract_somatic_features/feature_collection.py ./extract_somatic_features/feature_collection.py
COPY ./extract_somatic_features/utils.py ./extract_somatic_features/utils.py
COPY ./extract_somatic_features/file_io.py ./extract_somatic_features/file_io.py
COPY ./extract_somatic_features/Fix_mesh.py ./extract_somatic_features/Fix_mesh.py
COPY ./extract_somatic_features/save_features.py ./extract_somatic_features/save_features.py