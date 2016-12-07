/*
  Pay Chess client v2.0 series, neural network
  Copyright (C) 2015  Lester Vecsey

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include <unistd.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "neuron.h"

int prep_neuron(neuron_t *neurons, long int num_neurons) {

  long int n;

  for (n = 0; n < num_neurons; n++) {
    neurons[n].input = 0.0;
    neurons[n].hidden = 0.0;
    neurons[n].output = 0.0;
  }
  
  return 0;
  
}

int initial_weights(neuron_t *neurons, long int num_neurons, int rnd_fd) {

  uint64_t rnds[2];
  long int num_rnds = sizeof(rnds) / sizeof(uint64_t);
  
  long int n;

  ssize_t bytes_read;
  
  for (n = 0; n < num_neurons; n++) {
    bytes_read = read(rnd_fd, rnds, sizeof(uint64_t) * num_rnds);
    if (bytes_read != sizeof(uint64_t) * num_rnds) return -1;
    neurons[n].weight1 = ((double) rnds[0]) - 9223372036854775808.0;
    neurons[n].weight2 = ((double) rnds[1]) - 9223372036854775808.0;
  }

  return 0;
  
}

