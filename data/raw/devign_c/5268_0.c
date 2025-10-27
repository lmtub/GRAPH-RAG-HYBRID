static void vc1_inv_trans_4x8_c(uint8_t *dest, int linesize, DCTELEM *block)

{

    int i;

    register int t1,t2,t3,t4,t5,t6,t7,t8;

    DCTELEM *src, *dst;

    const uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    src = block;

    dst = block;

    for(i = 0; i < 8; i++){

        t1 = 17 * (src[0] + src[2]) + 4;

        t2 = 17 * (src[0] - src[2]) + 4;

        t3 = 22 * src[1] + 10 * src[3];

        t4 = 22 * src[3] - 10 * src[1];



        dst[0] = (t1 + t3) >> 3;

        dst[1] = (t2 - t4) >> 3;

        dst[2] = (t2 + t4) >> 3;

        dst[3] = (t1 - t3) >> 3;



        src += 8;

        dst += 8;

    }



    src = block;

    for(i = 0; i < 4; i++){

        t1 = 12 * (src[ 0] + src[32]) + 64;

        t2 = 12 * (src[ 0] - src[32]) + 64;

        t3 = 16 * src[16] +  6 * src[48];

        t4 =  6 * src[16] - 16 * src[48];



        t5 = t1 + t3;

        t6 = t2 + t4;

        t7 = t2 - t4;

        t8 = t1 - t3;



        t1 = 16 * src[ 8] + 15 * src[24] +  9 * src[40] +  4 * src[56];

        t2 = 15 * src[ 8] -  4 * src[24] - 16 * src[40] -  9 * src[56];

        t3 =  9 * src[ 8] - 16 * src[24] +  4 * src[40] + 15 * src[56];

        t4 =  4 * src[ 8] -  9 * src[24] + 15 * src[40] - 16 * src[56];



        dest[0*linesize] = cm[dest[0*linesize] + ((t5 + t1) >> 7)];

        dest[1*linesize] = cm[dest[1*linesize] + ((t6 + t2) >> 7)];

        dest[2*linesize] = cm[dest[2*linesize] + ((t7 + t3) >> 7)];

        dest[3*linesize] = cm[dest[3*linesize] + ((t8 + t4) >> 7)];

        dest[4*linesize] = cm[dest[4*linesize] + ((t8 - t4 + 1) >> 7)];

        dest[5*linesize] = cm[dest[5*linesize] + ((t7 - t3 + 1) >> 7)];

        dest[6*linesize] = cm[dest[6*linesize] + ((t6 - t2 + 1) >> 7)];

        dest[7*linesize] = cm[dest[7*linesize] + ((t5 - t1 + 1) >> 7)];



        src ++;

        dest++;

    }

}
