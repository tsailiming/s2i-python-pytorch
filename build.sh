#!/bin/bash

rm -rf /tmp/mnist/*
mkdir -p /tmp/mnist

(cd mnist && s2i build . pytorch:1.6.0 --as-dockerfile=/tmp/mnist/Dockerfile)
(cd /tmp/mnist && buildah bud -f . -t pytorch-mnist:l.0)
