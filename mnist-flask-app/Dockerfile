FROM registry.access.redhat.com/ubi8/python-39

LABEL Version 1.0

MAINTAINER Tsai Li Ming <tsai.liming@gmail.com>

USER 0 

ADD mnist-flask-app/app.py /tmp/src/
ADD mnist-flask-app/mnist/net.py /tmp/src/mnist/net.py
ADD mnist-flask-app/requirements.txt /tmp/src/
ADD mnist-flask-app/model/* /tmp/src/model/
RUN /usr/bin/fix-permissions /tmp/src

USER 1001
 
RUN /usr/libexec/s2i/assemble 
 
CMD /usr/libexec/s2i/run