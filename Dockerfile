FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir cupy-cuda90==4.2.0 chainer==4.2.0 chainercv==0.9.0

RUN pip3 install --no-cache-dir \
	tensorflow-gpu \
	keras==2.1.5 \
	gensim==3.4.0 \
	nltk==3.2.5 \
	pandas==0.22.0 \
	tables==3.4.3 \
	parse \
	jupyterlab

EXPOSE 8888


