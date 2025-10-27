static void float_to_int16_stride_altivec(int16_t *dst, const float *src,

                                          long len, int stride)

{

    int i, j;

    vector signed short d, s;



    for (i = 0; i < len - 7; i += 8) {

        d = float_to_int16_one_altivec(src + i);

        for (j = 0; j < 8; j++) {

            s = vec_splat(d, j);

            vec_ste(s, 0, dst);

            dst += stride;

        }

    }

}
