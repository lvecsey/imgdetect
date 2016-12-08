
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <sys/mman.h>
#include <errno.h>

#include "pixel.h"
#include "image.h"

long int kernel_identity[9] = {
  0, 0, 0
  ,0, 1, 0
  ,0, 0, 0
};

long int kernel_edge1[9] = {
  1, 0, -1
  ,0, 0, 0
  ,-1, 0, 1
};

long int kernel_edge2[9] = {
  0, 1, 0
  ,1, -4, 1
  ,0, 1, 0
};

long int kernel_edge3[9] = {
  -1, -1, -1
  ,-1, 8, -1
  ,-1, -1, -1
};

long int kernel_sharpen[9] = {
  0, -1, 0
  ,-1, 5, -1
  ,0 , -1, 0
};

long int kernel_boxblur[9] = {
  1, 1, 1
  ,1, 1, 1
  ,1, 1, 1
};

double boxblur_sf = 1.0 / 9.0;

long int kernel_unsharp[25] = {
  1, 4, 6, 4, 1
  ,4, 16, 24, 16, 4
  ,6, 24, 470, 24, 6
  ,4, 16, 24, 16, 4
  ,1, 4, 6, 4, 1
};

double unsharp_sf = 1.0 / 256.0;

#include "kernel_apply.h"

int kernel_apply(pixel_t *rgb, long int *kernel, double sf, long int xres, long int yres, long int channel, image_t *output) {

  long int xpos, ypos;
  long int xi, yi;

  long int xw = 3, yh = 3;
  
  image_t img = { .xres = xres, .yres = yres, .rgb = rgb };

  long int accum;

  if (kernel == kernel_unsharp) {
    xw = 5, yh = 5;
  }
  
  for (ypos = 0; ypos < yres; ypos++) {
    for (xpos = 0; xpos < xres; xpos++) {

      accum = 0;

      for (yi = 0; yi < yh; yi++) {
	for (xi = 0; xi < xw; xi++) {

	  if (ypos+yi < 1 || xpos+xi < 1) continue;
	  if (ypos+yi >= yres-2 || ypos+yi >= xres-2) continue;

	  switch(channel) {
	  case CHAN_RED: accum += kernel[yi*3+xi] * img.rgb[ (ypos+yi-1)*xres+xpos+xi-1 ].r; break;
	  case CHAN_GREEN: accum += kernel[yi*3+xi] * img.rgb[ (ypos+yi-1)*xres+xpos+xi-1 ].g; break;
	  case CHAN_BLUE: accum += kernel[yi*3+xi] * img.rgb[ (ypos+yi-1)*xres+xpos+xi-1 ].b; break;
	  }

	}
      }

      accum *= sf;
      
      output->rgb[ ypos*xres+xpos ].r = accum;
      output->rgb[ ypos*xres+xpos ].g = accum;
      output->rgb[ ypos*xres+xpos ].b = accum;

    }
  }

  return 0;

}

