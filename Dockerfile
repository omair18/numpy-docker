
  
FROM ubuntu:18.04

WORKDIR /tmp

RUN apt-get update && apt-get install -y python3-dev python3-pip python3-setuptools
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR    /workspace
CMD ["/bin/sh"]