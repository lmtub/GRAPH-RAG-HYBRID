static int pix_norm1_altivec(uint8_t *pix, int line_size)

{

    int i, s = 0;

    const vector unsigned int zero =

        (const vector unsigned int) vec_splat_u32(0);

    vector unsigned int sv = (vector unsigned int) vec_splat_u32(0);

    vector signed int sum;



    for (i = 0; i < 16; i++) {

        /* Read the potentially unaligned pixels. */

        //vector unsigned char pixl = vec_ld(0,  pix);

        //vector unsigned char pixr = vec_ld(15, pix);

        //vector unsigned char pixv = vec_perm(pixl, pixr, perm);

        vector unsigned char pixv = vec_vsx_ld(0,  pix);



        /* Square the values, and add them to our sum. */

        sv = vec_msum(pixv, pixv, sv);



        pix += line_size;

    }

    /* Sum up the four partial sums, and put the result into s. */

    sum = vec_sums((vector signed int) sv, (vector signed int) zero);

    sum = vec_splat(sum, 3);

    vec_vsx_st(sum, 0, &s);

    return s;

}
