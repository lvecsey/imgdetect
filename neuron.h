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

#ifndef NEURON_H
#define NEURON_H

typedef struct {
  double input;
  double weight1;
  double hidden;
  double weight2;
  double output;
  double expected;
} neuron_t;

int prep_neuron(neuron_t *neurons, long int num_neurons);

int initial_weights(neuron_t *neurons, long int num_neurons, int rnd_fd);

#endif
