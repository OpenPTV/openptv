/*
Definition for the tracking frame buffers. Each frame holds target information
for all cameras, correspondence information and path links information.
*/

#ifndef TRACKING_FRAME_BUF_H
#define TRACKING_FRAME_BUF_H

#define POSI 80
#define STR_MAX_LEN 255

typedef struct
{
  int     pnr;
  double  x, y;
  int     n, nx, ny, sumg;
  int     tnr;
}
target;

int compare_targets(target *t1, target *t2);
int read_targets(target buffer[], char* file_base, int frame_num);
int write_targets(target buffer[], int num_targets, char* file_base, \
    int frame_num);

typedef struct
{
  int nr;
  int p[4];
}
corres;

int compare_corres(corres *c1, corres *c2);
#define CORRES_NONE -1

typedef double coord_t;
typedef float fitness_t;

typedef struct Pstruct
{
  coord_t x[3]; /*coordinates*/
  int prev, next; /*pointer to prev or next link*/
  int prio; /*Prority of link is used for differen levels*/
  fitness_t decis[POSI]; /*Bin for decision critera of possible links to next dataset*/
  fitness_t finaldecis; /*final decision critera by which the link was established*/
  int linkdecis[POSI]; /* pointer of possible links to next data set*/
  int inlist; /* Counter of number of possible links to next data set*/
} P;

int compare_path_info(P *p1, P *p2);
void register_link_candidate(P *self, fitness_t fitness, int cand);
#define PREV_NONE -1
#define NEXT_NONE -2
#define PRIO_DEFAULT 2 
void reset_links(P *self);

int read_path_frame(corres *cor_buf, P *path_buf, \
    char *corres_file_base, char *linkage_file_base, 
    char *prio_file_base, int frame_num);
int write_path_frame(corres *cor_buf, P *path_buf, int num_parts,\
    char *corres_file_base, char *linkage_file_base, 
    char *prio_file_base, int frame_num);

typedef struct {
    P *path_info;
    corres *correspond;
    target **targets;
    int num_cams, max_targets;
    int num_parts; /* Number of 3D particles in the correspondence buffer */
    int *num_targets; /* Pointer to array of 2D particle counts per image. */
} frame;

void frame_init(frame *new_frame, int num_cams, int max_targets);
void free_frame(frame *self);
int read_frame(frame *self, char *corres_file_base, char *linkage_file_base,
    char *prio_file_base, char **target_file_base, int frame_num);
int write_frame(frame *self, char *corres_file_base, char *linkage_file_base,
    char *prio_file_base, char **target_file_base, int frame_num);


typedef struct {
    /* _ring_vec is the underlying double-size vector, buf is the pointer to 
    the start of the ring. */
    frame **buf, **_ring_vec;
    int buf_len, num_cams;
    char *corres_file_base, *linkage_file_base, *prio_file_base;
    char **target_file_base;
} framebuf;

void fb_init(framebuf *new_buf, int buf_len, int num_cams, int max_targets,\
    char *corres_file_base, char* linkage_file_base, char *prio_file_base,
    char **target_file_base);
void fb_free(framebuf *self);
void fb_next(framebuf *self);
void fb_prev(framebuf *self);
int fb_read_frame_at_end(framebuf *self, int frame_num, int read_links);
int fb_write_frame_from_start(framebuf *self, int frame_num);

#endif
