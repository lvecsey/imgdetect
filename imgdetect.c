
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <dirent.h>
#include <libgen.h>

#include <sys/mman.h>

#include "neuron.h"

#include <pixel.h>
#include <image.h>
#include <sigmoid.h>

#include "neur2.h"
#include "weight.h"

#include "kernel_apply.h"

double sigmoid_deriv(double x) {
  double sig = sigmoid(x);
  return sig * (1.0 - sig);
}

typedef struct {

  long int num_input;
  long int num_hidden;
  long int num_output;

  double learning_rate;
  double alpha_momentum;
  neur2_t *inputs;
  neur2_t *hidden;  
  weight_t *weights1;
  weight_t *weights2;
  neur2_t output;
  double expected;

} imgdetect_t;

struct pdir {

  char filename[240];
  double expected;

  struct pdir *next;
  
};

long int count_items(struct pdir *base) {

  struct pdir *p = base;
  
  long int counter = 0;

  while (p != NULL) {

    counter++;
    
    p = p->next;

  }

  return counter;

}

#define NUM_ITERS 20

double square(double x) { return x * x; }

int show_inputsummary(imgdetect_t *id, long int span);
int show_weightpack1(imgdetect_t *id, long int span);
int show_gradient1(imgdetect_t *id, long int span); 
int show_hiddensummary(imgdetect_t *id, long int span);
int show_weightpack2(imgdetect_t *id, long int span);
int show_gradient2(imgdetect_t *id, long int span); 
int show_outputsummary(imgdetect_t *id, long int span);
int show_expected(imgdetect_t *id, long int span);

int show_avg(imgdetect_t *id);

double get_expected(char *str) {

  if (!strcmp(str, "motorcycle")) return 0.0;

  if (!strcmp(str, "faces")) return 1.0;

  return 0.5;
  
}

int set_expected(neuron_t *neurons, long int num_neurons, char *str) {

  long int n;

  double expected = get_expected(str);

  for (n = 0; n < num_neurons; n++) {
    neurons[n].expected = expected;
  }

  return 0;
  
}

int repaint_inputs(imgdetect_t *id, pixel_t *rgb, long int num_pixels) {

  long int n;

  for (n = 0; n < id->num_input; n++) {

    id->inputs[n].sum = 0.5 * rgb[n*num_pixels/id->num_input].r / 65535.0;
    id->inputs[n].output = sigmoid(id->inputs[n].sum);
    
  }

  return 0;
  
}

int rerun_network(imgdetect_t *id) {

  double sum;

  long int j, i;
  
  for (j = 0; j < id->num_hidden; j++) {

    sum = 0.0;
    for (i = 0; i < id->num_input; i++) { 
  
      sum += id->inputs[i].output * id->weights1[j*id->num_input+i].weight;

    }

    id->hidden[j].sum = sum;
    id->hidden[j].output = sigmoid(sum);
    
  }

  sum = 0;
  for (i = 0; i < id->num_hidden; i++) {

    sum += id->hidden[i].output * id->weights2[i].weight;

  }

  id->output.sum = sum;
  id->output.output = sigmoid(sum);

  return 0;

}

int plot_mse(uint8_t *gray, long int xres, long int yres, double mse, long int iterno, long int num_updates) {

  double y;
  long int xpos, ypos;

  y = mse;
	
  xpos = (iterno * xres) / num_updates;
  ypos = y * (yres>>1); ypos += yres>>1;

  if (xpos < 0) xpos = -xpos;
  if (ypos < 0) ypos = -ypos;

  xpos %= (xres-1);
  ypos %= (yres-1);
  
  gray[ypos*xres/8+xpos/8] &= ~(1<<(7-(xpos%8)));

  return 0;

}

int setprevious_deltas(imgdetect_t *id) {

  long int n;

  id->output.previous_delta = id->output.delta;

  for (n = 0; n < id->num_hidden; n++) {
    id->hidden[n].previous_delta = id->hidden[n].delta;
  }

  for (n = 0; n < id->num_input; n++) {
    id->inputs[n].previous_delta = id->inputs[n].delta;
  }

  return 0;
  
}

double calc_sumA(imgdetect_t *id, long int j, weight_t *weights) {

  long int i;
  
  double sum = 0.0;
  
  for (i = 0; i < id->num_input; i++) {
  
    sum += weights[j*id->num_input+i].weight;

  }

  return sum;
  
}


double calc_sumB(imgdetect_t *id, long int i, weight_t *weights) {

  long int j;
    
  double sum = 0.0;
  
  for (j = 0; j < id->num_hidden; j++) {
  
    sum += weights[j*id->num_input+i].weight;

  }

  return sum;
  
}

int run_training(imgdetect_t *id, double *mse, uint8_t *gray, long int xres, long int yres) {

  double sum;

  double sderiv_output;

  long int j, i;

  double y;

  long int xpos, ypos;

  double E;
  double a, ideal;

  double delta;
  double gradient;
  
  ideal = id->expected;
  a = id->output.output;

  E = a - ideal;
  
  sderiv_output = sigmoid_deriv(id->output.sum);

  delta = -E * sderiv_output;

  for (i = 0; i < id->num_hidden; i++) {

    id->hidden[i].delta = sigmoid_deriv(id->hidden[i].sum) * id->weights2[i].weight * delta;

    gradient = sigmoid(id->hidden[i].sum) * delta;
    
    id->weights2[i].weight += id->learning_rate * gradient + id->alpha_momentum * id->hidden[i].previous_delta;

    id->weights2[i].gradient = gradient;
    
    y = id->weights2[i].weight;
    y = sigmoid(6.0 * y - 3.0);
      
    xpos = i * xres / id->num_hidden;
    ypos = y * (yres>>1); ypos += yres>>1;

    gray[ypos*xres/8+xpos/8] &= ~(1<<(7-(xpos%8)));
    
  }

  for (j = 0; j < id->num_hidden; j++) {

    sum = calc_sumB(id, i, id->weights1);
    id->hidden[j].delta = sigmoid_deriv(id->hidden[j].sum) * sum * delta;
    
  }
  
  for (j = 0; j < id->num_hidden; j++) {
	  
    for (i = 0; i < id->num_input; i++) {

      gradient = sigmoid(id->hidden[j].sum) * id->output.delta;

      id->weights1[j*id->num_input+i].weight += id->learning_rate * gradient + id->alpha_momentum * id->hidden[j].previous_delta;

      id->weights1[j*id->num_input+i].gradient = gradient;
      
      y = id->weights1[j*id->num_input+i].weight;
      y = sigmoid(6.0 * y - 3.0);
      
      xpos = (j*id->num_input+i) * xres / (id->num_hidden * id->num_input);
      ypos = y * (yres>>1); ypos += yres>>1;

      gray[ypos*xres/8+xpos/8] &= ~(1<<(7-(xpos%8)));
      
    }
	  
  }

  id->output.delta = delta;

  setprevious_deltas(id);
  
  return 0;

}

int collect_dir(struct pdir *base, char *dirname, double expected) {

  DIR *d;
  struct dirent *p;

  struct pdir *entry;
  
  d = opendir(dirname);
  if (d == NULL) {
    perror("opendir");
    return -1;
  }

  for (;;) {
    p = readdir(d);
    if (p==NULL) break;

    if (p->d_name[0] == '.' && p->d_name[1] == 0) continue;
    if (p->d_name[0] == '.' && p->d_name[1] == '.' && p->d_name[2] == 0) continue;    

    if (!p->d_name[0]) return -1;
    
    printf("Collecting %s/%s for processing as expected=%g\n", dirname, p->d_name, expected);

    entry = malloc(sizeof(struct pdir));
    if (entry==NULL) {
      perror("malloc");
      return -1;
    }

    entry->next = base->next;
    sprintf(entry->filename, "%s/%s", dirname, p->d_name);
    entry->expected = expected;

    base->next = entry;

  }
  
  return 0;

}

int process_dir(struct pdir *base, imgdetect_t *id, double *mse, uint8_t *gray, long int xres, long int yres) {

  int fd;

  struct pdir *entry;
  
  pixel_t *rgb;
  long int num_pixels;

  struct stat buf;
  int retval;

  void *m;

  long int input_xres, input_yres;
  
  long int fileno = 0;

  image_t output;
  
  double sum;
  
  entry = base->next;

  sum = 0.0;
  
  while (entry != NULL) {
    
    printf("Opening %s for processing.. \n", entry->filename);

    retval = sscanf(strrchr(entry->filename, '_'), "_%ldx%ld.rgb", &input_xres, &input_yres);

    if (retval != 2) {
      printf("Warning, retval for sscanf was %d\n", retval);
      entry = entry->next;
      continue;
    }
      
    fd = open(entry->filename, O_RDWR);
    if (fd == -1) {
      perror("open");
      return -1;
    }

    retval = fstat(fd, &buf);
    if (retval == -1) {
      perror("fstat");
      return -1;
    }
    num_pixels = buf.st_size / sizeof(pixel_t);

    m = mmap(NULL, buf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
      perror("mmap");
      return -1;
    }

    rgb = malloc(buf.st_size);
    if (rgb==NULL) { perror("malloc"); return -1; }
    
    output.rgb = rgb;
    output.xres = input_xres;
    output.yres = input_yres;
    retval = kernel_apply(m, kernel_edge1, 0.0, input_xres, input_yres, CHAN_RED, &output);

    close(fd);
    munmap(m, buf.st_size);
    
    id->expected = entry->expected;
    
    repaint_inputs(id, rgb, num_pixels);
    
    free(rgb);

    rerun_network(id);

    sum += square(id->expected - id->output.output);
    
    run_training(id, mse, gray, xres, yres);
    
    entry = entry->next;
    
    fileno++;
    
  }

  *mse = sum / fileno;

  printf("MSE %.02g%%\n", 100.0 * mse[0]);

  {
    long int span = 7;
    show_inputsummary(id, span);
    show_weightpack1(id, span);
    show_gradient1(id, span);
    show_hiddensummary(id, span);
    show_weightpack2(id, span);    
    show_gradient2(id, span);
    show_outputsummary(id, span);
    show_expected(id, span);
  }
  
  return 0;
  
}

int test_filename(imgdetect_t *id, char *filename) {

  int fd;

  pixel_t *rgb;
  long int num_pixels;

  long int input_xres, input_yres;
  
  struct stat buf;
  int retval;

  void *m;

  image_t output;
  
  printf("Testing %s against neural network.. \n", filename);

  retval = sscanf(strrchr(filename, '_'), "_%ldx%ld.rgb", &input_xres, &input_yres);

  if (retval != 2) {
    printf("Warning, retval for sscanf was %d\n", retval);
    return -1;
  }
  
  fd = open(filename, O_RDWR);
  if (fd == -1) {
    perror("open");
    return -1;
  }
  
  retval = fstat(fd, &buf);
  if (retval==-1) {
    perror("fstat");
    return -1;
  }
  num_pixels = buf.st_size / sizeof(pixel_t);

  m = mmap(NULL, buf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) {
    perror("mmap");
    return -1;
  }

  rgb = malloc(buf.st_size);
  if (rgb==NULL) { perror("malloc"); return -1; }
    
  output.rgb = rgb;
  output.xres = input_xres;
  output.yres = input_yres;
  retval = kernel_apply(m, kernel_edge1, 0.0, input_xres, input_yres, CHAN_RED, &output);

  close(fd);
  munmap(m, buf.st_size);
  
  repaint_inputs(id, rgb, num_pixels);

  free(rgb);
  
  rerun_network(id);

  show_inputsummary(id, 10);
  show_weightpack1(id, 10);
  show_gradient1(id, 10);
  show_hiddensummary(id, 10);
  show_weightpack2(id, 10);
  show_gradient2(id, 10);
  show_outputsummary(id, 10);
  
  printf("Matched a %s\n", id->output.output >= 0.80 ? "face" : "motorcycle");
  
  return 0;
  
}

int show_weightpack1(imgdetect_t *id, long int span) {

  long int j;
  
  long int i;
  
  long int n;

  long int jspan_incr = id->num_hidden / span;
  
  for (j = 0; j < id->num_hidden; j+=jspan_incr) {
    for (i = 0; i < span; i++) {
      n = i * id->num_input / span;
      printf("%g ", id->weights1[j*id->num_input+n].weight);
    }
  }

  putchar('\n');

  return 0;
  
}

int show_gradient1(imgdetect_t *id, long int span) {

  long int j;
  
  long int i;
  
  long int n;

  long int jspan_incr = id->num_hidden / span;
  
  for (j = 0; j < id->num_hidden; j+=jspan_incr) {
    for (i = 0; i < span; i++) {
      n = i * id->num_input / span;
      printf("%g ", id->weights1[j*id->num_input+n].gradient);
    }
  }

  putchar('\n');

  return 0;
  
}

int show_weightpack2(imgdetect_t *id, long int span) {

  long int j = 0;

  long int n;
  
  for (j = 0; j < span; j++) {
    n = j * id->num_hidden / span;
    printf("%g ", id->weights2[n].weight);
  }

  putchar('\n');

  return 0;
  
}

int show_gradient2(imgdetect_t *id, long int span) {

  long int j = 0;

  long int n;
  
  for (j = 0; j < span; j++) {
    n = j * id->num_hidden / span;
    printf("%g ", id->weights2[n].gradient);
  }

  putchar('\n');

  return 0;
  
}

int show_outputsummary(imgdetect_t *id, long int span) {

  printf("output: %g\n", id->output.output);

  return 0;
  
}

int show_expected(imgdetect_t *id, long int span) {

  printf("expected: %g\n", id->expected);

  return 0;
  
}


int show_inputsummary(imgdetect_t *id, long int span) {

  long int j = 0;;

  long int n;

  printf("input: ");
  
  for (j = 0; j < span; j++) {
    n = j * id->num_input / span;
    printf("%g ", id->inputs[n].output);
  }

  putchar('\n');

  return 0;
  
}

int show_hiddensummary(imgdetect_t *id, long int span) {

  long int j = 0;;

  long int n;

  printf("hidden: ");
  
  for (j = 0; j < span; j++) {
    n = j * id->num_hidden / span;
    printf("%g ", id->hidden[n].output);
  }

  putchar('\n');

  return 0;
  
}

int show_avg(imgdetect_t *id) {

  printf("avg=%g\n", id->output.output);
  
  return 0;

}

int main(int argc, char *argv[]) {

  long int xres = 1600, yres = 900;
  long int num_pixels = xres * yres;
  size_t img_sz = num_pixels >> 3;
  uint8_t *gray;

  long int n;
  
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd;
  void *m;
  
  char *filename = argc>1 ? argv[1] : NULL;
  
  int rnd_fd = open("/dev/urandom", O_RDONLY);
  
  char *dir1 = "traindata/motorcycle";
  char *dir2 = "traindata/faces";

  imgdetect_t id = { .learning_rate = 0.15, .alpha_momentum = 0.475, .output.output = 0.5 };
  
  int retval;

  ssize_t bytes_read;

  struct pdir *base = malloc(sizeof(struct pdir));

  enum { NONE, LARGE_NN, MEDIUM_NN, SMALL_NN };

  long int neuralnet_size = SMALL_NN;
  
  if (neuralnet_size == LARGE_NN) {
    id.num_input = 4903;
    id.num_hidden = 4357;
  }
  else if (neuralnet_size == MEDIUM_NN) { 
    id.num_input = 1481;
    id.num_hidden = 1109;
  }
  else if (neuralnet_size == SMALL_NN) {
    id.num_input = 149;
    id.num_hidden = 127;
  }
  
  id.num_output = 1;

  id.inputs = calloc(id.num_input, sizeof(neur2_t));
  id.hidden = calloc(id.num_hidden, sizeof(neur2_t));
  
  id.weights1 = calloc(id.num_input * id.num_hidden, sizeof(weight_t));
  id.weights2 = calloc(id.num_hidden * id.num_output, sizeof(weight_t));

  if (id.weights1 == NULL || id.weights2 == NULL) {
    perror("calloc");
    return -1;
  }
  
  {
    uint64_t rnd;

    for (n = 0; n < id.num_input * id.num_hidden; n++) {
      bytes_read = read(rnd_fd, &rnd, sizeof(uint64_t));
      if (bytes_read != sizeof(uint64_t)) return -1;
      id.weights1[n].weight = 3.0 * rnd / 18446744073709551615.0 - 1.5;
    }

    for (n = 0; n < id.num_hidden * id.num_output; n++) {
      bytes_read = read(rnd_fd, &rnd, sizeof(uint64_t));
      if (bytes_read != sizeof(uint64_t)) return -1;
      id.weights2[n].weight = 3.0 * rnd / 18446744073709551615.0 - 1.5;
    }
  }
  
  {
    char strbuf[240];
    sprintf(strbuf, "errweights_%ldx%ld.gray", xres, yres);
    fd = open(strbuf, O_RDWR | O_CREAT, mode);
    retval = ftruncate(fd, img_sz);
    m = mmap(NULL, img_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
      perror("mmap");
      return -1;
    }
    gray = (uint8_t*) m;
  }
  
  for (n = 0; n < img_sz; n++) gray[n] = 255;

  memset(base->filename, 0, sizeof(base->filename));
  base->expected = 0.0;
  base->next = NULL;

  retval = collect_dir(base, dir1, get_expected("motorcycle"));
  if (retval==-1) {
    return -1;
  }

  retval = collect_dir(base, dir2, get_expected("faces"));
  if (retval==-1) {
    return -1;
  }

  {
    
    double mse = 1.0;

    for (n = 0; n < NUM_ITERS; n++) {
    
      retval = process_dir(base, &id, &mse, gray, xres, yres);

      plot_mse(gray, xres, yres, mse, n, NUM_ITERS);
      
    }

  }
      
  printf("Avg output value should be 0.0 if matching a motorcycle\n");
  printf("And 1.0 if matching a face.\n");
  
  test_filename(&id, filename); 

  show_avg(&id);
  
  return 0;

}
