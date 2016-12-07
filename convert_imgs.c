#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <dirent.h>
#include <libgen.h>

int main(int argc, char *argv[]) {

  DIR *d;
  
  struct dirent *p;

  char *src_dir = argc>1 ? argv[1] : NULL;
  char *dst_basedir = argc>2 ? argv[2] : NULL;

  char cmd[240];
  
  d = opendir(src_dir);

  if (d==NULL) {
    perror("opendir");
    return -1;
  }

  for (;;) {
    p = readdir(d);
    if (p==NULL) break;

    printf("%s\n", p->d_name);

    if (p->d_name[0] == '.' && p->d_name[1] == 0) continue;
    if (p->d_name[0] == '.' && p->d_name[1] == '.' && p->d_name[2] == 0) continue;    
    
    sprintf(cmd, "identify %s/%s", src_dir, p->d_name);

    {

      FILE *fp = popen(cmd, "r");
      char *line = NULL;
      size_t len = 0;
      ssize_t bytes_read;

      int retval;
      
      while ((bytes_read = getline(&line, &len, fp)) != -1) {
	if (len > 0) {
	  char filename[240];
	  char filetype[8];
	  char res[8];
	  retval = sscanf(line, "%s %s %s", filename, filetype, res);
	  printf("[%s] %s\n", res, p->d_name);

	  sprintf(cmd, "convert %s/%s -depth 16 %s/%s_%s.rgb", src_dir, p->d_name, dst_basedir, basename(p->d_name), res);
	  retval = system(cmd);
	  if (0 != retval) {
	    fprintf(stderr, "%s: Trouble with convert call.\n", __FUNCTION__);
	    return -1;
	  }
	}
      }

      pclose(fp);
      
    }
    
  }
  
  closedir(d);
  
  return 0;

}
  
