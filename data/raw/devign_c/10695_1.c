static inline void FUNC(idctSparseColAdd)(pixel *dest, int line_size,

                                          DCTELEM *col)

{

    int a0, a1, a2, a3, b0, b1, b2, b3;

    INIT_CLIP;



    IDCT_COLS;



    dest[0] = CLIP(dest[0] + ((a0 + b0) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a1 + b1) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a2 + b2) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a3 + b3) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a3 - b3) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a2 - b2) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a1 - b1) >> COL_SHIFT));

    dest += line_size;

    dest[0] = CLIP(dest[0] + ((a0 - b0) >> COL_SHIFT));

}
