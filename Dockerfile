FROM python:3.5
MAINTAINER omair
#RUN apt-get update && apt-get -y install ipython3
#liblapack-dev libatlas-dev \gfortran \libhdf5-dev \libnetcdf-dev

RUN pip install  numpy pipenv
COPY 	  . /app
WORKDIR    /app
RUN 	pipenv install --deploy --dev
ENV        SHELL=/bin/bash
ENTRYPOINT ["pipenv", "run"]
CMD ["python"]
#CMD 	   ["/bin/sh"]
#ENTRYPOINT ["/bin/sh -c"]
#CMD        ["ls"]
