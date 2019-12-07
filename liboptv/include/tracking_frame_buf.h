/*
Definition for the tracking frame buffers. Each frame holds target information
for all cameras, correspondence information and path links information.
*/

#ifndef TRACKING_FRAME_BUF_H
#define TRACKING_FRAME_BUF_H

/* For point positions */
#include "vec_utils.h"

#define POSI 80
#define STR_MAX_LEN 255

#define PT_UNUSED -999

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

typedef float fitness_t;

typedef struct Pstruct
{
  vec3d x; /*coordinates*/
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


/*
 * Following is the frame buffer class, that holds a given number of frame structs,
 * and treats them as a deque (doube-ended queue, a queue that can advance forward 
 * or backward).
 * 
 * The memory locations are advanced with fb_next()/fb_prev(). Filling out the new
 * frame that joined the queue and flushing out the frame that exited the queue 
 * are respectively done using fb_read_frame_at_end() and fb_write_frame_from_start().
 * 
 * To make it possible to read frames from different sources, we use a 
 * base-class/child-class structure. framebuf_base is the base class, and implements
 * only the memory-advance methods. The filling-out of frames is done by 
 * "virtual" functions. That is to say, fb_read_frame_at_end, for example,
 * only forwards the call to a function that really reads the data.
 * 
 * The virtual functions are placed in the "vtable" by the constructor - the 
 * function that allocates and populates the child class. 
 * 
 * In tracking_framebuf.c and here there's an example child-class that reads 
 * frame information from *_target files. This is the original mode of 
 * liboptv. The struct framebuf *inherits* from framebuf_base (by incorporating
 * the base as 1st member - the order is important). fb_init(), the constructor,
 * then populates the vtable with e.g. fb_disk_* functions. These are the derived
 * implementations of the base-class virtual functions. 
 * 
 * There is also a virtual destructor, fb_free(), which delegates to the free() 
 * virtual function, and implemented in the example by e.g. fb_disk_free().
 * 
 * Note: fb_disk_free does not release the strings it holds, as I don't remember if
 * it owns them. 
 * 
 * Yes, in C++ it's easier :)
 */

// Virtual function table definitions for frame buffer objects:
typedef struct framebuf_base* fbp;
typedef struct {
    void (*free)(fbp self);
    int (*read_frame_at_end)(fbp self, int frame_num, int read_links);
    int (*write_frame_from_start)(fbp self, int frame_num);
} fb_vtable;

typedef struct framebuf_base {
    fb_vtable *_vptr;
    
    /* _ring_vec is the underlying double-size vector, buf is the pointer to 
    the start of the ring. */
    frame **buf, **_ring_vec;
    int buf_len, num_cams;
} framebuf_base;

// These just call the corresponding virtual function. 
// Actual implementations are below each child class.
void fb_free(framebuf_base *self);
int fb_read_frame_at_end(framebuf_base *self, int frame_num, int read_links);
int fb_write_frame_from_start(framebuf_base *self, int frame_num);

// Non-virtual methods of the base class.
void fb_base_init(framebuf_base *new_buf, int buf_len, int num_cams, int max_targets);
void fb_next(framebuf_base *self);
void fb_prev(framebuf_base *self);

// child class that reads from _target files.
typedef struct {
    framebuf_base base; // must be 1st member.
    
    char *corres_file_base, *linkage_file_base, *prio_file_base;
    char **target_file_base;
} framebuf;

void fb_init(framebuf *new_buf, int buf_len, int num_cams, int max_targets,
    char *corres_file_base, char* linkage_file_base, char *prio_file_base,
    char **target_file_base);

void fb_disk_free(framebuf_base *self);
int fb_disk_read_frame_at_end(framebuf_base *self, int frame_num, int read_links);
int fb_disk_write_frame_from_start(framebuf_base *self, int frame_num);

// Child class that reads from memory. The incoming/outgoing frames are read
// or written to a memory in address changed from outside on each frame. The 
// tracking code is responsible for managing those pointers. The information
// on when to change them is passed by using callbacks: whenever the framebuf
// finishes a read or write it calls the respective callback, to let the 
// tracker do the necessary updates.
typedef int (*mem_io_fun)(int frame_num, void* tracker_info);
typedef struct {
    framebuf_base base; // must be 1st member.
    
    frame **incoming, **outgoing; 
    mem_io_fun read_callback;
    mem_io_fun write_callback;
    void* tracker_info; // constant pointe, passed to the callbacks.
    
    // pointer to pointer: the outside pointer is constant, and describes 
    // where the driver code keeps the address of the (changing) incoming
    // frame, or the possibly-changing place to write the outgoing frame.
    // Note that the incoming/outgoing `frame` struct is copied from/to
    // incoming/outgoing mmemory, but the subordinate allocations (corres, 
    // target, etc.) are not - it's up to you to recycle or delete them,
    // just like it's up to you to allocate them.
} framebuf_inmem;

void fb_inmem_init(
    framebuf_inmem *new_buf, 
    int buf_len, int num_cams, int max_targets,
    frame **incoming, frame **outgoing, void* tracker_info,
    mem_io_fun read_callback, mem_io_fun write_callback
);

void fb_inmem_free(framebuf_base *self);
// Note: frame number is not currently used. When we fold the threaded 
// code into here, we might have an objects that can translate frame-num to 
// pointer, instead of using **incoming etc. Anyway, we must have this
// argument because that's how the base-class defines this virtual function.
// read_links is ignored because either the incoming frame has them or it doesn't.
int fb_inmem_read_frame_at_end(framebuf_base *self, int frame_num, int read_links);
int fb_inmem_write_frame_from_start(framebuf_base *self, int frame_num);

#endif
