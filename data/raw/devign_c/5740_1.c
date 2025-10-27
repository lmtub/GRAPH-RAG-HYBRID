static int pix_sum_altivec(uint8_t *pix, int line_size)

{

    int i, s;

    const vector unsigned int zero =

        (const vector unsigned int) vec_splat_u32(0);

    vector unsigned int sad = (vector unsigned int) vec_splat_u32(0);

    vector signed int sumdiffs;



    for (i = 0; i < 16; i++) {

        /* Read the potentially unaligned 16 pixels into t1. */

        //vector unsigned char pixl = vec_ld(0,  pix);

        //vector unsigned char pixr = vec_ld(15, pix);

        //vector unsigned char t1   = vec_perm(pixl, pixr, perm);

        vector unsigned char t1   = vec_vsx_ld(0,  pix);



        /* Add each 4 pixel group together and put 4 results into sad. */

        sad = vec_sum4s(t1, sad);



        pix += line_size;

    }



    /* Sum up the four partial sums, and put the result into s. */

    sumdiffs = vec_sums((vector signed int) sad, (vector signed int) zero);

    sumdiffs = vec_splat(sumdiffs, 3);

    vec_vsx_st(sumdiffs, 0, &s);

    return s;

}
