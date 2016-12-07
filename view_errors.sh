#!/bin/bash

# display -size 1600x900 -depth 1 ./errweights_1600x900.gray
convert -size 1600x900 -depth 1 ./errweights_1600x900.gray errweights.png

feh --fullscreen errweights.png
