static inline void scale_mv(AVSContext *h, int *d_x, int *d_y,

                            cavs_vector *src, int distp)

{

    int den = h->scale_den[src->ref];



    *d_x = (src->x * distp * den + 256 + (src->x >> 31)) >> 9;

    *d_y = (src->y * distp * den + 256 + (src->y >> 31)) >> 9;

}
