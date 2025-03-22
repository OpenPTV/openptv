# Vector utilities definitions for import in to other Cython files.

cdef extern from "optv/vec_utils.h":
    ctypedef double vec3d[3]
    
    int vec_cmp(vec3d vec1, vec3d vec2)
    void vec_copy(vec3d dest, vec3d src)

