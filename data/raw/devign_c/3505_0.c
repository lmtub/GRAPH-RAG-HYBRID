void ff_vp3_idct_put_altivec(uint8_t *dst, int stride, DCTELEM block[64])

{

    vec_u8 t;

    IDCT_START



    // pixels are signed; so add 128*16 in addition to the normal 8

    vec_s16 v2048 = vec_sl(vec_splat_s16(1), vec_splat_u16(11));

    eight = vec_add(eight, v2048);



    IDCT_1D(NOP, NOP)

    TRANSPOSE8(b0, b1, b2, b3, b4, b5, b6, b7);

    IDCT_1D(ADD8, SHIFT4)



#define PUT(a)\

    t = vec_packsu(a, a);\

    vec_ste((vec_u32)t, 0, (unsigned int *)dst);\

    vec_ste((vec_u32)t, 4, (unsigned int *)dst);



    PUT(b0)     dst += stride;

    PUT(b1)     dst += stride;

    PUT(b2)     dst += stride;

    PUT(b3)     dst += stride;

    PUT(b4)     dst += stride;

    PUT(b5)     dst += stride;

    PUT(b6)     dst += stride;

    PUT(b7)

}
