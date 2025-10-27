static void vp8_idct_add_c(uint8_t *dst, DCTELEM block[16], ptrdiff_t stride)

{

    int i, t0, t1, t2, t3;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    DCTELEM tmp[16];



    for (i = 0; i < 4; i++) {

        t0 = block[0*4+i] + block[2*4+i];

        t1 = block[0*4+i] - block[2*4+i];

        t2 = MUL_35468(block[1*4+i]) - MUL_20091(block[3*4+i]);

        t3 = MUL_20091(block[1*4+i]) + MUL_35468(block[3*4+i]);

        block[0*4+i] = 0;

        block[1*4+i] = 0;

        block[2*4+i] = 0;

        block[3*4+i] = 0;



        tmp[i*4+0] = t0 + t3;

        tmp[i*4+1] = t1 + t2;

        tmp[i*4+2] = t1 - t2;

        tmp[i*4+3] = t0 - t3;

    }



    for (i = 0; i < 4; i++) {

        t0 = tmp[0*4+i] + tmp[2*4+i];

        t1 = tmp[0*4+i] - tmp[2*4+i];

        t2 = MUL_35468(tmp[1*4+i]) - MUL_20091(tmp[3*4+i]);

        t3 = MUL_20091(tmp[1*4+i]) + MUL_35468(tmp[3*4+i]);



        dst[0] = cm[dst[0] + ((t0 + t3 + 4) >> 3)];

        dst[1] = cm[dst[1] + ((t1 + t2 + 4) >> 3)];

        dst[2] = cm[dst[2] + ((t1 - t2 + 4) >> 3)];

        dst[3] = cm[dst[3] + ((t0 - t3 + 4) >> 3)];

        dst += stride;

    }

}
