Visualizing the inner workings of the `trackcorr_c_loop` in `track.c` with a 3D matplotlib plot requires a mechanism to send data from the C environment to Python at various points within the loop. This typically involves defining a C data structure to hold the visualization information, a callback function pointer in C, and implementing a Cython wrapper that allows a Python function to be called from C.

Here's a conceptual guide on how to modify `track.c` and `tracking.pyx`:

**Overall Strategy:**

1.  **Define Data Structures (C):** Create C structs to hold the 3D coordinates, IDs, scores, and other relevant data for the current particle, its predictions, and candidates at different stages.
2.  **Introduce a Callback Mechanism (C & Cython):**
    *   In C, define a function pointer type for a callback that will be invoked with the visualization data.
    *   Add an instance of this function pointer to the `tracking_run` struct.
    *   In Cython (`tracking.pyx`), create a C-callable Python function wrapper. This wrapper will take the C data, convert it into Python objects (dictionaries, lists), and then call a user-provided Python plotting function.
    *   Modify the `Tracker` class in `tracking.pyx` to accept a Python callback function during initialization and set it up for the C side.
3.  **Populate and Invoke Callback (C):** In `trackcorr_c_loop` (and potentially `trackback_c`), at key decision points or after important calculations, populate the C data structure and call the registered callback function if it's set.
4.  **Implement Visualization (Python):** The user (you) will write a Python function that takes the data from the callback and uses `matplotlib` to render the 3D plot. This function will handle clearing the plot and drawing new points, lines, and annotations for each step.

---

**Step 1: Modify C code (`track.c` and a new header e.g., `visualization.h`)**

**1.1. Create `visualization.h` (or add to an existing relevant header like `track.h`):**
This file will define the data structures for visualization and the callback type.

```c
// visualization.h
#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "vec3d.h" // Assuming vec3d is defined elsewhere (e.g. in OpenPTV's linalg or similar)

#define MAX_VIS_POINTS 20 // Maximum number of candidate points to send for visualization at once
#define VIS_MSG_SIZE 256

// A structure to represent a point for visualization
typedef struct {
    vec3d pos;
    int id;         // e.g., ftnr or particle index
    double val1;    // Generic value (e.g., frequency, score, metric)
    double val2;    // Another generic value
    char label[32]; // Short label for the point
} Vis_Point_t;

// Main data structure passed to the visualization callback
typedef struct {
    int current_step;           // Current frame/step number
    int current_particle_idx;   // Index 'h' of the particle in fb->buf[1]

    // Stage description
    char stage_message[VIS_MSG_SIZE];

    // Core particle positions
    Vis_Point_t P0_prev;        // X[0] - Previous position of current particle's track
    Vis_Point_t P1_current;     // X[1] - Current particle being tracked
    int P0_exists;

    // Predictions and search centers
    Vis_Point_t P2_search_vol_center; // X[2] - Predicted center for 1st set of candidates (w)
    Vis_Point_t P5_search_vol_center; // X[5] - Predicted center for 2nd set of candidates (wn)

    // Candidate lists
    int num_candidates_W;       // Number of 'w' candidates (from P2)
    Vis_Point_t candidates_W[MAX_VIS_POINTS]; // List of X[3]s

    int num_candidates_WN;      // Number of 'wn' candidates (from P5)
    Vis_Point_t candidates_WN[MAX_VIS_POINTS]; // List of X[4]s
    
    int active_W_candidate_idx_in_list; // if processing a specific W, its index in candidates_W
    int active_WN_candidate_idx_in_list; // if processing a specific WN, its index in candidates_WN


    // Link quality metrics (when a specific link is being evaluated)
    double link_angle;
    double link_acc;
    double link_dl;         // distance metric part
    double link_rr;         // overall "goodness"
    int link_quali;

    // Flag if a new particle was added in this sub-step
    int particle_added_flag;
    Vis_Point_t added_particle_info;

} VisualizationData_t;

// Callback function pointer type
typedef void (*VisualizationCallback_t)(VisualizationData_t* vis_data);

#endif // VISUALIZATION_H
```

**1.2. Modify `tracking_run.h`:**
Add the callback pointer and a buffer for the visualization data to the `tracking_run` struct.

```c
// tracking_run.h
// ... other includes ...
#include "visualization.h" // Add this

typedef struct tracking_run {
    // ... existing fields ...
    framebuf_base *fb;
    Calibration **cal;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    sequence_par *seq_par; // Ensure this is present
    double lmax, ymin, ymax; // Ensure these are present
    long npart, nlinks; // Ensure these are present
    
    // New fields for visualization
    VisualizationCallback_t vis_callback;
    VisualizationData_t vis_data_buffer; // Buffer to hold data for a single callback
} tracking_run;

// Update tr_new prototype if you add parameters to it for the callback
// tracking_run* tr_new(sequence_par* seq_par, track_par* tpar, ... , VisualizationCallback_t callback);
// Or, add a separate function to set the callback:
void tr_set_visualization_callback(tracking_run* tr, VisualizationCallback_t callback);
```

**1.3. Modify `track.c`:**

*   **Include `visualization.h`**.
*   **Update `tr_new` or add `tr_set_visualization_callback`**:
    ```c
    // In tr_new, if adding as a parameter:
    // run->vis_callback = callback; // if passed as argument
    // else initialize:
    // run->vis_callback = NULL;

    // Or, add a new function:
    void tr_set_visualization_callback(tracking_run* tr, VisualizationCallback_t callback) {
        if (tr) {
            tr->vis_callback = callback;
        }
    }
    ```
*   **Modify `trackcorr_c_loop`**: This is the most complex part. You need to populate `run_info->vis_data_buffer` and call `run_info->vis_callback` at various stages.

    ```c
    // track.c
    #include "visualization.h" // Make sure it's included

    // Helper to clear and initialize vis_data_buffer for a new particle h
    void init_vis_data(VisualizationData_t* vd, int step, int h_idx) {
        memset(vd, 0, sizeof(VisualizationData_t)); // Clear all fields
        vd->current_step = step;
        vd->current_particle_idx = h_idx;
        vd->active_W_candidate_idx_in_list = -1;
        vd->active_WN_candidate_idx_in_list = -1;
    }

    // Helper to populate a Vis_Point_t
    void populate_vis_point(Vis_Point_t* vp, vec3d pos, int id, double v1, double v2, const char* label_fmt, ...) {
        va_list args;
        vec_copy(vp->pos, pos);
        vp->id = id;
        vp->val1 = v1;
        vp->val2 = v2;
        va_start(args, label_fmt);
        vsnprintf(vp->label, sizeof(vp->label)-1, label_fmt, args);
        va_end(args);
    }


    void trackcorr_c_loop (tracking_run *run_info, int step) {
        // ... existing variables ...
        VisualizationData_t* vis_buf = &run_info->vis_data_buffer; // Shortcut

        // ...
        for (h = 0; h < orig_parts; h++) {
            if (run_info->vis_callback) {
                init_vis_data(vis_buf, step, h);
            }

            // ...
            vec_copy(X[1], curr_path_inf->x);
            if (run_info->vis_callback) {
                populate_vis_point(&vis_buf->P1_current, X[1], h, 0, 0, "P1 (h=%d)", h);
            }

            if (curr_path_inf->prev >= 0) {
                // ...
                vec_copy(X[0], ref_path_inf->x); // X[0] is prev_pos
                search_volume_center_moving(X[0], X[1], X[2]); // X[2] is predicted
                if (run_info->vis_callback) {
                    vis_buf->P0_exists = 1;
                    populate_vis_point(&vis_buf->P0_prev, X[0], curr_path_inf->prev, 0, 0, "P0 (prev=%d)", curr_path_inf->prev);
                    populate_vis_point(&vis_buf->P2_search_vol_center, X[2], -1, 0, 0, "P2 (from X0,X1)");
                    snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d: X[0], X[1] known. Predicted X[2].", step, h);
                    run_info->vis_callback(vis_buf);
                }
            } else {
                vec_copy(X[2], X[1]); // X[2] = X[1] if no prev
                 if (run_info->vis_callback) {
                    vis_buf->P0_exists = 0;
                    populate_vis_point(&vis_buf->P2_search_vol_center, X[2], -1, 0, 0, "P2 (no X0, =X1)");
                    snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d: No X[0]. X[2]=X[1].", step, h);
                    run_info->vis_callback(vis_buf);
                }
            }

            w = sorted_candidates_in_volume(X[2], v1, fb->buf[2], run_info);
            if (run_info->vis_callback && w != NULL) {
                vis_buf->num_candidates_W = 0;
                for (int i = 0; w[i].ftnr != TR_UNUSED && i < MAX_VIS_POINTS; ++i) {
                    P* cand_w_path_info = &(fb->buf[2]->path_info[w[i].ftnr]);
                    populate_vis_point(&vis_buf->candidates_W[i], cand_w_path_info->x, w[i].ftnr, w[i].freq, 0, "W[%d] ftnr=%d", i, w[i].ftnr);
                    vis_buf->num_candidates_W++;
                }
                snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d: Found %d W-candidates for X[2].", step, h, vis_buf->num_candidates_W);
                run_info->vis_callback(vis_buf);
                vis_buf->num_candidates_W = 0; // Clear for next callback unless it's cumulative
            }
            if (w == NULL) continue;


            mm = 0;
            while (w[mm].ftnr != TR_UNUSED) {
                vis_buf->active_W_candidate_idx_in_list = mm < MAX_VIS_POINTS ? mm : -1;

                ref_path_inf = &(fb->buf[2]->path_info[w[mm].ftnr]);
                vec_copy(X[3], ref_path_inf->x); // X[3] is current W candidate pos

                // ... calculate X[5] ...
                if (curr_path_inf->prev >= 0) { /* ... X[5] = ... */ } else { /* ... X[5] = ... */ }

                if (run_info->vis_callback) {
                    // (Re-populate P0, P1, P2 if needed or assume they are still set)
                    if (mm < MAX_VIS_POINTS) { // Only if it was added to vis_buf->candidates_W
                         // Add X[3] specifically for this active W candidate
                         populate_vis_point(&vis_buf->candidates_W[0], X[3], w[mm].ftnr, w[mm].freq, 0, "Active W (ftnr=%d)", w[mm].ftnr);
                         vis_buf->num_candidates_W = 1; // Show only the active one
                    } else {
                        vis_buf->num_candidates_W = 0;
                    }
                    populate_vis_point(&vis_buf->P5_search_vol_center, X[5], -1, 0, 0, "P5 (from X1,X3)");
                    angle_acc(X[1], X[2], X[3], &angle0, &acc0); // metrics for X[1]->X[3] vs X[2]
                    vis_buf->link_angle = angle0; vis_buf->link_acc = acc0;

                    snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d: W-cand ftnr %d (X[3]). Predicted X[5]. Angle0=%.1f, Acc0=%.1e", step, h, w[mm].ftnr, angle0, acc0);
                    run_info->vis_callback(vis_buf);
                    // Clear W, P5, metrics for next iteration stage if needed
                    vis_buf->num_candidates_W = 0; 
                    memset(&vis_buf->P5_search_vol_center, 0, sizeof(Vis_Point_t));
                }

                wn = sorted_candidates_in_volume(X[5], v1, fb->buf[3], run_info);
                if (run_info->vis_callback && wn != NULL) {
                    vis_buf->num_candidates_WN = 0;
                     for (int i = 0; wn[i].ftnr != TR_UNUSED && i < MAX_VIS_POINTS; ++i) {
                        P* cand_wn_path_info = &(fb->buf[3]->path_info[wn[i].ftnr]);
                        populate_vis_point(&vis_buf->candidates_WN[i], cand_wn_path_info->x, wn[i].ftnr, wn[i].freq, 0, "WN[%d] ftnr=%d",i, wn[i].ftnr);
                        vis_buf->num_candidates_WN++;
                    }
                    snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d, W-cand ftnr %d: Found %d WN-candidates for X[5].", step, h, w[mm].ftnr, vis_buf->num_candidates_WN);
                    run_info->vis_callback(vis_buf);
                    vis_buf->num_candidates_WN = 0; // Clear for next callback
                }


                if (wn != NULL) {
                    kk = 0;
                    while (wn[kk].ftnr != TR_UNUSED) {
                        vis_buf->active_WN_candidate_idx_in_list = kk < MAX_VIS_POINTS ? kk : -1;
                        ref_path_inf = &(fb->buf[3]->path_info[wn[kk].ftnr]);
                        vec_copy(X[4], ref_path_inf->x); // X[4] is current WN candidate

                        // ... calculations for angle1, acc1, acc, angle, quali, rr ...
                        angle_acc(X[3], X[4], X[5], &angle1, &acc1);
                        // ...
                        acc = (acc0+acc1)/2; angle = (angle0+angle1)/2;
                        quali = wn[kk].freq+w[mm].freq;
                        // ... rr calculation ... dl = ...; rr = ...;

                        if (run_info->vis_callback) {
                            // (Re-populate P0,P1,P2,P5, active W if needed)
                            if (kk < MAX_VIS_POINTS) {
                                populate_vis_point(&vis_buf->candidates_WN[0], X[4], wn[kk].ftnr, wn[kk].freq, 0, "Active WN (ftnr=%d)", wn[kk].ftnr);
                                vis_buf->num_candidates_WN = 1; // Show only active one
                            } else {
                                vis_buf->num_candidates_WN = 0;
                            }
                            vis_buf->link_angle = angle; vis_buf->link_acc = acc;
                            vis_buf->link_dl = dl; vis_buf->link_rr = rr; vis_buf->link_quali = quali;

                            snprintf(vis_buf->stage_message, VIS_MSG_SIZE, "Step %d, Ptc %d, W_ftnr %d, WN_ftnr %d (X[4]). Eval: A=%.1f, Acc=%.1e, RR=%.2e, Q=%d", step, h, w[mm].ftnr, wn[kk].ftnr, angle, acc, rr, quali);
                            run_info->vis_callback(vis_buf);
                             vis_buf->num_candidates_WN = 0; // Clear WN
                        }
                        // ... (link registration logic) ...
                        kk++;
                    } // End WN loop
                    vis_buf->active_WN_candidate_idx_in_list = -1;
                }
                free(wn);

                // ... logic for "creating new particle position" using X[5] ...
                // if (quali >= 2) ... dl = point_position ... X[4] ... angle_acc(X[3], X[4], X[5]...)
                // if (tpar->add) { add_particle(fb->buf[3], X[4], philf); }
                // if callback needed for particle addition:
                if (run_info->vis_callback && tpar->add /* && particle was added */) {
                    // Populate vis_buf->particle_added_flag = 1;
                    // Populate vis_buf->added_particle_info with X[4] data.
                    // snprintf(vis_buf->stage_message, ... "Added new particle from X[5]/X[4] eval");
                    // run_info->vis_callback(vis_buf);
                    // vis_buf->particle_added_flag = 0;
                }

                mm++;
            } // End W loop
            vis_buf->active_W_candidate_idx_in_list = -1;
            free(w);
            // ... other logic like "inlist still zero" and particle addition ...
            // Add callbacks similarly for those stages if desired.
        } // End h-loop
    }
    ```
    **Note:** The above callback placements are illustrative. You'll need to refine them based on exactly what state you want to see. Each call to `run_info->vis_callback(vis_buf)` will send the current state of `vis_buf` to Python. Be mindful of clearing parts of `vis_buf` (like candidate lists) if they are not meant to be cumulative for that specific callback.

---

**Step 2: Modify Cython code (`tracking.pyx`)**

```python
# tracking.pyx
import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport free, malloc # For potential memory management if needed
from libc.string cimport memset # For clearing structs

# Assuming vec3d is double[3]
ctypedef double vec3d[3]

# --- C definitions from visualization.h ---
cdef extern from "visualization.h":
    int MAX_VIS_POINTS
    int VIS_MSG_SIZE

    ctypedef struct Vis_Point_t:
        vec3d pos
        int id
        double val1
        double val2
        char label[32]

    ctypedef struct VisualizationData_t:
        int current_step
        int current_particle_idx
        char stage_message[VIS_MSG_SIZE]
        Vis_Point_t P0_prev
        Vis_Point_t P1_current
        int P0_exists
        Vis_Point_t P2_search_vol_center
        Vis_Point_t P5_search_vol_center
        int num_candidates_W
        Vis_Point_t candidates_W[MAX_VIS_POINTS]
        int num_candidates_WN
        Vis_Point_t candidates_WN[MAX_VIS_POINTS]
        int active_W_candidate_idx_in_list
        int active_WN_candidate_idx_in_list
        double link_angle
        double link_acc
        double link_dl
        double link_rr
        int link_quali
        int particle_added_flag
        Vis_Point_t added_particle_info

    ctypedef void (*VisualizationCallback_t)(VisualizationData_t* vis_data)

# --- C definitions from tracking_run.h and tracker.pxd ---
# Make sure tracking_run struct is defined here or imported if it's in tracker.pxd
# And tr_set_visualization_callback if you added it
cdef extern from "optv/tracker.h": # Or wherever tr_new etc are defined
    ctypedef struct tracking_run:
        pass # Add actual fields if needed by cython, or rely on opaque pointer
    
    tracking_run* tr_new(...) # Existing signature
    void track_forward_start(tracking_run* tr)
    void trackcorr_c_loop(tracking_run* tr, int step)
    void trackcorr_c_finish(tracking_run* tr, int step)
    # If you added tr_set_visualization_callback:
    void tr_set_visualization_callback(tracking_run* tr, VisualizationCallback_t callback)
    # ... any other C functions you use ...


# Global variable to store the Python callback
python_visualization_handler = None

# C function that will be called from C code. This wraps the Python call.
cdef void c_py_vis_callback_wrapper(VisualizationData_t* c_data_ptr):
    global python_visualization_handler
    if python_visualization_handler is not None and c_data_ptr is not None:
        # Convert C_data_ptr to a Python dictionary
        py_data = {}
        py_data['current_step'] = c_data_ptr.current_step
        py_data['current_particle_idx'] = c_data_ptr.current_particle_idx
        py_data['stage_message'] = c_data_ptr.stage_message.decode('utf-8', errors='replace')
        
        def convert_vis_point(c_vp):
            return {
                'pos': tuple(c_vp.pos), # Convert C array to tuple
                'id': c_vp.id,
                'val1': c_vp.val1,
                'val2': c_vp.val2,
                'label': c_vp.label.decode('utf-8', errors='replace')
            }

        py_data['P0_exists'] = bool(c_data_ptr.P0_exists)
        if py_data['P0_exists']:
            py_data['P0_prev'] = convert_vis_point(c_data_ptr.P0_prev)
        py_data['P1_current'] = convert_vis_point(c_data_ptr.P1_current)
        py_data['P2_search_vol_center'] = convert_vis_point(c_data_ptr.P2_search_vol_center)
        py_data['P5_search_vol_center'] = convert_vis_point(c_data_ptr.P5_search_vol_center)

        py_data['candidates_W'] = [convert_vis_point(c_data_ptr.candidates_W[i]) for i in range(c_data_ptr.num_candidates_W)]
        py_data['candidates_WN'] = [convert_vis_point(c_data_ptr.candidates_WN[i]) for i in range(c_data_ptr.num_candidates_WN)]
        
        py_data['active_W_candidate_idx_in_list'] = c_data_ptr.active_W_candidate_idx_in_list
        py_data['active_WN_candidate_idx_in_list'] = c_data_ptr.active_WN_candidate_idx_in_list

        py_data['link_metrics'] = {
            'angle': c_data_ptr.link_angle,
            'acc': c_data_ptr.link_acc,
            'dl': c_data_ptr.link_dl,
            'rr': c_data_ptr.link_rr,
            'quali': c_data_ptr.link_quali,
        }
        
        py_data['particle_added_flag'] = bool(c_data_ptr.particle_added_flag)
        if py_data['particle_added_flag']:
            py_data['added_particle_info'] = convert_vis_point(c_data_ptr.added_particle_info)

        # Call the actual Python handler
        python_visualization_handler(py_data)


cdef class Tracker:
    # ... existing __init__ and other methods ...
    def __init__(self, ControlParams cpar, VolumeParams vpar, 
        TrackingParams tpar, SequenceParams spar, list cals,
        dict naming=None, flatten_tol=0.0001, object py_vis_callback=None): # Added py_vis_callback
        
        self._keepalive = (cpar, vpar, tpar, spar, cals)
        
        if naming is None:
            naming = default_naming
        else:
            naming = {k: _encode_if_needed(v) for k, v in naming.items()}
            for key in default_naming:
                if key not in naming:
                    naming[key] = default_naming[key]
        
        self.run_info = tr_new(spar._sequence_par, tpar._track_par,
            vpar._volume_par, cpar._control_par, TR_BUFSPACE, MAX_TARGETS,
            naming['corres'], naming['linkage'], naming['prio'], 
            cal_list2arr(cals), flatten_tol)
        
        # Set up visualization callback
        global python_visualization_handler
        python_visualization_handler = py_vis_callback
        
        if self.run_info != NULL and py_vis_callback is not None:
            # If using tr_set_visualization_callback:
            tr_set_visualization_callback(self.run_info, c_py_vis_callback_wrapper)
            # If vis_callback is a field directly set (requires C struct def in Cython):
            # self.run_info.vis_callback = c_py_vis_callback_wrapper # This might need casting or careful struct def
            print("Python visualization callback has been set.")
        elif py_vis_callback is not None:
             print("Warning: run_info is NULL, cannot set visualization callback.")


    # ... other methods like restart, step_forward, finalize ...

    def __dealloc__(self):
        if self.run_info is not NULL:
            # ... existing free calls ...
            # No specific free for vis_data_buffer as it's part of run_info
            # Reset global handler if appropriate, though instance-specific handler is better
            global python_visualization_handler
            python_visualization_handler = None # Or manage this more carefully if multiple Trackers exist
        # ...
```

---

**Step 3: Python Visualization Code (Example)**

This is the Python function you would pass to the `Tracker`'s constructor.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class TrackingVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion() # Interactive mode on
        self.fig.show()
        self.text_annotations = []

    def plot_point(self, point_data, color, marker='o', size=50, label_prefix=""):
        if not point_data or not isinstance(point_data.get('pos'), tuple) or len(point_data['pos']) != 3:
            # print(f"Skipping invalid point data: {point_data}")
            return
        pos = np.array(point_data['pos'])
        self.ax.scatter(pos[0], pos[1], pos[2], c=color, marker=marker, s=size, label=f"{label_prefix}{point_data.get('label', '')}")
        # self.text_annotations.append(self.ax.text(pos[0], pos[1], pos[2], f"ID:{point_data.get('id', 'N/A')}\nV1:{point_data.get('val1',0):.2f}", size=8))


    def update_plot(self, vis_data):
        self.ax.clear() # Clear previous plot
        for ann in self.text_annotations: # Clear previous text
            ann.remove()
        self.text_annotations = []

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        title = f"Step: {vis_data['current_step']}, Particle: {vis_data['current_particle_idx']}\nStage: {vis_data['stage_message']}"
        self.ax.set_title(title, fontsize=10)

        # Plot P1 (current particle)
        if 'P1_current' in vis_data and vis_data['P1_current'].get('pos'):
            self.plot_point(vis_data['P1_current'], 'red', marker='X', size=100, label_prefix="P1: ")

        # Plot P0 (previous particle)
        if vis_data.get('P0_exists', False) and 'P0_prev' in vis_data and vis_data['P0_prev'].get('pos'):
            self.plot_point(vis_data['P0_prev'], 'pink', marker='x', size=80, label_prefix="P0: ")
            # Draw line from P0 to P1
            p0 = np.array(vis_data['P0_prev']['pos'])
            p1 = np.array(vis_data['P1_current']['pos'])
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', alpha=0.7)


        # Plot P2 (search center for W)
        if 'P2_search_vol_center' in vis_data and vis_data['P2_search_vol_center'].get('pos'):
            self.plot_point(vis_data['P2_search_vol_center'], 'blue', marker='s', size=60, label_prefix="P2: ")

        # Plot W candidates
        for i, cand_w in enumerate(vis_data.get('candidates_W', [])):
            is_active = (i == vis_data.get('active_W_candidate_idx_in_list', -1) and vis_data['active_W_candidate_idx_in_list'] != -1)
            self.plot_point(cand_w, 'cyan' if not is_active else 'darkcyan', marker='o', size=40 if not is_active else 60, label_prefix="W: ")
            if vis_data['P2_search_vol_center'].get('pos'): # Draw line from P2 to W candidate
                 p2 = np.array(vis_data['P2_search_vol_center']['pos'])
                 w_pos = np.array(cand_w['pos'])
                 self.ax.plot([p2[0], w_pos[0]], [p2[1], w_pos[1]], [p2[2], w_pos[2]], 'b:', alpha=0.5)


        # Plot P5 (search center for WN)
        if 'P5_search_vol_center' in vis_data and vis_data['P5_search_vol_center'].get('pos'):
            self.plot_point(vis_data['P5_search_vol_center'], 'green', marker='^', size=60, label_prefix="P5: ")
            # If an active W candidate X[3] exists, draw line from X[3] to X[5]
            active_w_idx = vis_data.get('active_W_candidate_idx_in_list', -1)
            if active_w_idx != -1 and active_w_idx < len(vis_data.get('candidates_W',[])):
                active_w_data = vis_data['candidates_W'][active_w_idx]
                if active_w_data and active_w_data.get('pos'):
                    x3_pos = np.array(active_w_data['pos'])
                    x5_pos = np.array(vis_data['P5_search_vol_center']['pos'])
                    self.ax.plot([x3_pos[0], x5_pos[0]], [x3_pos[1], x5_pos[1]], [x3_pos[2], x5_pos[2]], 'g--', alpha=0.7)


        # Plot WN candidates
        for i, cand_wn in enumerate(vis_data.get('candidates_WN', [])):
            is_active = (i == vis_data.get('active_WN_candidate_idx_in_list',-1) and vis_data['active_WN_candidate_idx_in_list']!=-1)
            self.plot_point(cand_wn, 'lime' if not is_active else 'darkgreen', marker='P', size=40 if not is_active else 60, label_prefix="WN: ")
            if vis_data['P5_search_vol_center'].get('pos'): # Draw line from P5 to WN candidate
                 p5 = np.array(vis_data['P5_search_vol_center']['pos'])
                 wn_pos = np.array(cand_wn['pos'])
                 self.ax.plot([p5[0], wn_pos[0]], [p5[1], wn_pos[1]], [p5[2], wn_pos[2]], 'g:', alpha=0.5)

        # Display link metrics
        metrics = vis_data.get('link_metrics', {})
        metrics_text = f"Link Eval: Angle={metrics.get('angle',0):.1f}, Acc={metrics.get('acc',0):.1e}\n" \
                       f"dl={metrics.get('dl',0):.2f}, RR={metrics.get('rr',0):.2e}, Q={metrics.get('quali',0)}"
        self.text_annotations.append(self.ax.text2D(0.02, 0.98, metrics_text, transform=self.ax.transAxes, ha='left', va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7)))

        # Added particle
        if vis_data.get('particle_added_flag', False) and 'added_particle_info' in vis_data:
            self.plot_point(vis_data['added_particle_info'], 'purple', marker='*', size=150, label_prefix="ADDED: ")
        
        # Auto-scale axes - can be tricky, might need manual limits
        # self.ax.autoscale_view() # this might not work well for 3d if points are sparse
        all_x = [p['pos'][0] for p_list in [vis_data.get(k, []) for k in ['P0_prev','P1_current','P2_search_vol_center','P5_search_vol_center','candidates_W','candidates_WN', 'added_particle_info']] for p in (p_list if isinstance(p_list, list) else [p_list]) if p and isinstance(p.get('pos'), tuple)]
        all_y = [p['pos'][1] for p_list in [vis_data.get(k, []) for k in ['P0_prev','P1_current','P2_search_vol_center','P5_search_vol_center','candidates_W','candidates_WN', 'added_particle_info']] for p in (p_list if isinstance(p_list, list) else [p_list]) if p and isinstance(p.get('pos'), tuple)]
        all_z = [p['pos'][2] for p_list in [vis_data.get(k, []) for k in ['P0_prev','P1_current','P2_search_vol_center','P5_search_vol_center','candidates_W','candidates_WN', 'added_particle_info']] for p in (p_list if isinstance(p_list, list) else [p_list]) if p and isinstance(p.get('pos'), tuple)]

        if all_x and all_y and all_z: # Check if any points were plotted
            mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
            max_range = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z), 1.0) # Ensure max_range is at least 1
            self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        else: # Default view if no points
            self.ax.set_xlim(-5,5); self.ax.set_ylim(-5,5); self.ax.set_zlim(-5,5)


        self.ax.legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(0.85, 1))
        self.fig.canvas.draw_idle()
        plt.pause(0.01) # Pause to allow plot to update, adjust as needed

# How to use it:
# visualizer = TrackingVisualizer()
# tracker = Tracker(cpar, vpar, tpar, spar, cals, py_vis_callback=visualizer.update_plot)
# tracker.full_forward() # or tracker.step_forward() in a loop
```

**Compiling:**
You'll need to ensure your `setup.py` (or build system) correctly compiles the Cython code and links against the C object files (`track.o` and any other C files from your project). `visualization.h` should be in the include path.

**Important Considerations:**

*   **Performance:** Calling Python from C frequently can be slow. For very dense data or high-speed requirements, this might be a bottleneck. You can control the frequency of callbacks (e.g., only call every Nth particle or Nth step, or only for specific events).
*   **Data Complexity:** The `VisualizationData_t` struct can become complex. Start simple and add more fields as needed. The conversion in `c_py_vis_callback_wrapper` must match this struct.
*   **Memory Management in C for `vis_data_buffer`:** The `vis_data_buffer` is part of `tracking_run`, so it's allocated and freed with `tracking_run`. The `candidates_W` and `candidates_WN` arrays are fixed-size within it. If you were to use dynamically allocated arrays within `VisualizationData_t`, you'd need to manage their memory carefully (e.g., C allocates, Cython copies, C frees, or C allocates and Python callback is responsible for freeing via another C helper). The fixed-size array approach is simpler here.
*   **Error Handling:** Add checks for `NULL` pointers and handle potential errors in data conversion.
*   **Clarity of Visualization:** Too much information can clutter the plot. Decide what are the most critical aspects to visualize at each stage. The `stage_message` will be very helpful.
*   **Build System:** Make sure your Cython extension in `setup.py` lists all relevant C source files (like `track.c`) and include directories.

This detailed outline provides a solid foundation. The implementation will require careful coding and debugging, especially the C data population and the Cython conversion layer. Start with visualizing just a few key points (e.g., X[1], X[2], and the first W candidate) and gradually add more detail.