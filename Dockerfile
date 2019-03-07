FROM frolvlad/alpine-python-machinelearning

MAINTAINER omair
#RUN apt-get update && apt-get -y install ipython3
#liblapack-dev libatlas-dev \gfortran \libhdf5-dev \libnetcdf-dev

#RUN pip3 install  numpy pipenv
COPY 	  . /app
WORKDIR    /app
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN python3 -m site
RUN 	pipenv install --deploy --dev
ENV        SHELL=/bin/bash
#ENTRYPOINT ["pipenv", "run"]
#CMD ["python3"]
CMD 	   ["/bin/sh"]
#ENTRYPOINT ["/bin/sh -c"]
#CMD        ["ls"]
