#!/bin/bash

rm -rf traindata
mkdir -p traindata/faces
mkdir -p traindata/motorcycle
./convert_imgs ~/images/motorcycle traindata/motorcycle
./convert_imgs ~/images/faces traindata/faces

exit 0
