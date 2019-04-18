# A Dockerfile for an Ubuntu machine that can compile liboptv and the Cython extensions

FROM ubuntu:18.04

RUN apt-get update
RUN apt-get --assume-yes install cmake
RUN apt-get --assume-yes install g++
RUN apt-get --assume-yes install python-pip
RUN pip install virtualenv
RUN virtualenv /env --python=`which python2`
