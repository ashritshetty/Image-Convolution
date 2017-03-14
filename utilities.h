#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER 512

void read_image(char *name, unsigned char **image, int *im_width, int *im_height)
{
	FILE *fip;
	char buf[BUFFER];
	char *parse;
	int im_size;
	fip=fopen(name,"rb");
	if(fip==NULL)
	{
		fprintf(stderr,"ERROR:Cannot open %s\n",name);
		exit(0);
	}
	fgets(buf,BUFFER,fip);
	do
	{
		fgets(buf,BUFFER,fip);
	}
	while(buf[0]=='#');
	parse=strtok(buf," ");
	(*im_width)=atoi(parse);
	parse=strtok(NULL,"\n");
	(*im_height)=atoi(parse);
	fgets(buf,BUFFER,fip);
	parse=strtok(buf," ");
	im_size=(*im_width)*(*im_height);
	(*image)=(unsigned char *)malloc(sizeof(unsigned char)*im_size);
	fread(*image,1,im_size,fip);
	fclose(fip);
}

template<class T>
void read_image_template(char *name, T **image, int *im_width, int *im_height)
{
	unsigned char *temp_img;
	read_image(name, &temp_img, im_width, im_height);
  (*image) = (T*)malloc(sizeof(T)*(*im_width)*(*im_height));
	for(int i=0;i<(*im_width)*(*im_height);i++)
	{
		(*image)[i]=(T)temp_img[i];
	}
	free(temp_img);
}

void write_image(char *name, unsigned char *image, int im_width, int im_height)
{
	FILE *fop;
	int im_size=im_width*im_height;
	fop=fopen(name,"w+");
	fprintf(fop,"P5\n%d %d\n255\n",im_width,im_height);
	fwrite(image,sizeof(unsigned char),im_size,fop);
	fclose(fop);
}

template<class T>
void write_image_template(char *name, T *image, int im_width, int im_height)
{
	unsigned char *temp_img=(unsigned char*)malloc(sizeof(unsigned char)*im_width*im_height);
	for(int i=0;i<(im_width*im_height);i++)
	{
		temp_img[i]=(unsigned char)image[i];
	}
	write_image(name,temp_img,im_width,im_height);
	free(temp_img);
}

void read_matrix(char *filename, int *m, int **values)
{
  FILE* name;
  int i, t1, t2;
  name = fopen(filename, "r+");
  if(name != NULL)
  {
    fscanf(name, "%d\n", &t1);
    *m = t1;
    *values = (int *)calloc(t1, sizeof(int));
    for(i = 0; i < t1; i++)
    {
      fscanf(name, "%d ", &t2);
      *(*values+i) = t2;
    }
    fclose(name);
  }
  else
  {
    printf("File read failed\n");
    exit(1);
  }
}

void write_matrix(char *filename, int *m, int **values)
{
  FILE* name;
  int i, t1, t2;
  name = fopen(filename, "w+");
  if(name != NULL)
  {
    t1 = *m;
    fprintf(name, "%d\n", t1);
    for(i = 0; i < t1; i++)
    {
      t2 = *(*values+i);
      fprintf(name, "%d ", t2);
    }
    fclose(name);
  }
  else
  {
    printf("File write failed\n");
    exit(1);
  }
}
