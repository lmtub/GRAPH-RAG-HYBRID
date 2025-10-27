static void compute_antialias_float(MPADecodeContext *s,

                              GranuleDef *g)

{

    float *ptr;

    int n, i;



    /* we antialias only "long" bands */

    if (g->block_type == 2) {

        if (!g->switch_point)

            return;

        /* XXX: check this for 8000Hz case */

        n = 1;

    } else {

        n = SBLIMIT - 1;

    }



    ptr = g->sb_hybrid + 18;

    for(i = n;i > 0;i--) {

        float tmp0, tmp1;

        float *csa = &csa_table_float[0][0];

#define FLOAT_AA(j)\

        tmp0= ptr[-1-j];\

        tmp1= ptr[   j];\

        ptr[-1-j] = tmp0 * csa[0+4*j] - tmp1 * csa[1+4*j];\

        ptr[   j] = tmp0 * csa[1+4*j] + tmp1 * csa[0+4*j];



        FLOAT_AA(0)

        FLOAT_AA(1)

        FLOAT_AA(2)

        FLOAT_AA(3)

        FLOAT_AA(4)

        FLOAT_AA(5)

        FLOAT_AA(6)

        FLOAT_AA(7)



        ptr += 18;

    }

}
