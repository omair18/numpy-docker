
  
ARG CUDA_REPO=nvidia/cuda
ARG CUDA_VER=9.0
ARG CUDNN_VER=7
ARG UBUNTU_VER=16.04

FROM ${CUDA_REPO}:${CUDA_VER}-cudnn${CUDNN_VER}-devel-ubuntu${UBUNTU_VER}
ENV        SHELL=/bin/bash
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y python3-dev python3-pip python3-setuptools
#RUN pip3 install --trusted-host pypi.python.org pipenv pytest
RUN pip3 install pipenv pytest
RUN pip3 install numpy tensorflow_gpu==1.9.0
RUN pip3 install h5py keras
RUN apt-get update && apt-get install -y python3-tk
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH 
#python3 setup.py install 
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
WORKDIR    /app
COPY ./Pipfile /app
COPY ./Pipfile.lock /app
RUN pipenv install --deploy --dev

COPY      . /app
#ENTRYPOINT ["pipenv", "run"]
#CMD ["python3"]
CMD        ["/bin/sh"]
