#ifndef KERNEL_APPLY_H
#define KERNEL_APPLY_H

enum { CHAN_RED, CHAN_GREEN, CHAN_BLUE };

long int kernel_identify[9];
long int kernel_edge1[9];
long int kernel_edge2[9];
long int kernel_edge3[9];
long int kernel_sharpen[9];
long int kernel_boxblur[9];
long int kernel_gaussian[9];

double boxblur_sf;

long int kernel_unsharp[25];

double unsharp_sf;

int kernel_apply(pixel_t *rgb, long int *kernel, double sf, long int xres, long int yres, long int channel, image_t *output);

#endif
