
CC=gcc

LIBS=-lm

CFLAGS=-O3 -Wall -g -pg -I$(HOME)/src/libgxkit
LDFLAGS=-L$(HOME)/src/libgxkit

all : convert_imgs imgdetect

convert_imgs : convert_imgs.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

imgdetect : imgdetect.o neuron.o kernel_apply.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)
