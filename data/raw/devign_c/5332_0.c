static av_always_inline void filter_common(uint8_t *p, ptrdiff_t stride, int is4tap)

{

    LOAD_PIXELS

    int a, f1, f2;

    const uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    a = 3*(q0 - p0);



    if (is4tap)

        a += clip_int8(p1 - q1);



    a = clip_int8(a);



    // We deviate from the spec here with c(a+3) >> 3

    // since that's what libvpx does.

    f1 = FFMIN(a+4, 127) >> 3;

    f2 = FFMIN(a+3, 127) >> 3;



    // Despite what the spec says, we do need to clamp here to

    // be bitexact with libvpx.

    p[-1*stride] = cm[p0 + f2];

    p[ 0*stride] = cm[q0 - f1];



    // only used for _inner on blocks without high edge variance

    if (!is4tap) {

        a = (f1+1)>>1;

        p[-2*stride] = cm[p1 + a];

        p[ 1*stride] = cm[q1 - a];

    }

}
