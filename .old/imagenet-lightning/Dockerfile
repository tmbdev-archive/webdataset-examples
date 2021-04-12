FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -qqy curl
RUN ln -s /opt/conda/bin/python3 /usr/bin/python3
RUN ln -s /opt/conda/bin/conda /usr/bin/conda
RUN conda install pip
RUN ln -s /opt/conda/bin/pip /usr/bin/pip3
# RUN pip3 install torchvision
RUN pip3 install typer
RUN pip3 install pytorch-lightning
RUN pip3 install tensorboard
RUN pip3 install numpy scipy
RUN apt-get install -qqy git
RUN pip3 install jupyterlab
RUN pip3 install robustness
RUN apt-get install -qqy strace
ADD VERSION /VERSION
RUN pip3 install git+git://github.com/tmbdev/webdataset
