static inline void FUNC(idctSparseCol_extrashift)(int16_t *col)

#else

static inline void FUNC(idctSparseColPut)(pixel *dest, int line_size,

                                          int16_t *col)

{

    int a0, a1, a2, a3, b0, b1, b2, b3;



    IDCT_COLS;



    dest[0] = av_clip_pixel((a0 + b0) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a1 + b1) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a2 + b2) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a3 + b3) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a3 - b3) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a2 - b2) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a1 - b1) >> COL_SHIFT);

    dest += line_size;

    dest[0] = av_clip_pixel((a0 - b0) >> COL_SHIFT);

}



static inline void FUNC(idctSparseColAdd)(pixel *dest, int line_size,

                                          int16_t *col)

{

    int a0, a1, a2, a3, b0, b1, b2, b3;



    IDCT_COLS;



    dest[0] = av_clip_pixel(dest[0] + ((a0 + b0) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a1 + b1) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a2 + b2) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a3 + b3) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a3 - b3) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a2 - b2) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a1 - b1) >> COL_SHIFT));

    dest += line_size;

    dest[0] = av_clip_pixel(dest[0] + ((a0 - b0) >> COL_SHIFT));

}



static inline void FUNC(idctSparseCol)(int16_t *col)

#endif

{

    int a0, a1, a2, a3, b0, b1, b2, b3;



    IDCT_COLS;



    col[0 ] = ((a0 + b0) >> COL_SHIFT);

    col[8 ] = ((a1 + b1) >> COL_SHIFT);

    col[16] = ((a2 + b2) >> COL_SHIFT);

    col[24] = ((a3 + b3) >> COL_SHIFT);

    col[32] = ((a3 - b3) >> COL_SHIFT);

    col[40] = ((a2 - b2) >> COL_SHIFT);

    col[48] = ((a1 - b1) >> COL_SHIFT);

    col[56] = ((a0 - b0) >> COL_SHIFT);

}
