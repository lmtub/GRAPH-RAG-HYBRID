static int svq1_motion_inter_4v_block(MpegEncContext *s, GetBitContext *bitbuf,

                                      uint8_t *current, uint8_t *previous,

                                      int pitch, svq1_pmv *motion, int x, int y)

{

    uint8_t *src;

    uint8_t *dst;

    svq1_pmv mv;

    svq1_pmv *pmv[4];

    int i, result;



    /* predict and decode motion vector (0) */

    pmv[0] = &motion[0];

    if (y == 0) {

        pmv[1] =

        pmv[2] = pmv[0];

    } else {

        pmv[1] = &motion[(x / 8) + 2];

        pmv[2] = &motion[(x / 8) + 4];

    }



    result = svq1_decode_motion_vector(bitbuf, &mv, pmv);



    if (result != 0)

        return result;



    /* predict and decode motion vector (1) */

    pmv[0] = &mv;

    if (y == 0) {

        pmv[1] =

        pmv[2] = pmv[0];

    } else {

        pmv[1] = &motion[(x / 8) + 3];

    }

    result = svq1_decode_motion_vector(bitbuf, &motion[0], pmv);



    if (result != 0)

        return result;



    /* predict and decode motion vector (2) */

    pmv[1] = &motion[0];

    pmv[2] = &motion[(x / 8) + 1];



    result = svq1_decode_motion_vector(bitbuf, &motion[(x / 8) + 2], pmv);



    if (result != 0)

        return result;



    /* predict and decode motion vector (3) */

    pmv[2] = &motion[(x / 8) + 2];

    pmv[3] = &motion[(x / 8) + 3];



    result = svq1_decode_motion_vector(bitbuf, pmv[3], pmv);



    if (result != 0)

        return result;



    /* form predictions */

    for (i = 0; i < 4; i++) {

        int mvx = pmv[i]->x + (i  & 1) * 16;

        int mvy = pmv[i]->y + (i >> 1) * 16;



        // FIXME: clipping or padding?

        if (y + (mvy >> 1) < 0)

            mvy = 0;

        if (x + (mvx >> 1) < 0)

            mvx = 0;



        src = &previous[(x + (mvx >> 1)) + (y + (mvy >> 1)) * pitch];

        dst = current;



        s->dsp.put_pixels_tab[1][((mvy & 1) << 1) | (mvx & 1)](dst, src, pitch, 8);



        /* select next block */

        if (i & 1)

            current += 8 * (pitch - 1);

        else

            current += 8;

    }



    return 0;

}
