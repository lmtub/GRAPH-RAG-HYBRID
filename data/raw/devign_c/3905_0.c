static int svq1_motion_inter_block(MpegEncContext *s, GetBitContext *bitbuf,

                                   uint8_t *current, uint8_t *previous,

                                   int pitch, svq1_pmv *motion, int x, int y)

{

    uint8_t *src;

    uint8_t *dst;

    svq1_pmv mv;

    svq1_pmv *pmv[3];

    int result;



    /* predict and decode motion vector */

    pmv[0] = &motion[0];

    if (y == 0) {

        pmv[1] =

        pmv[2] = pmv[0];

    } else {

        pmv[1] = &motion[x / 8 + 2];

        pmv[2] = &motion[x / 8 + 4];

    }



    result = svq1_decode_motion_vector(bitbuf, &mv, pmv);



    if (result != 0)

        return result;



    motion[0].x         =

    motion[x / 8 + 2].x =

    motion[x / 8 + 3].x = mv.x;

    motion[0].y         =

    motion[x / 8 + 2].y =

    motion[x / 8 + 3].y = mv.y;



    if (y + (mv.y >> 1) < 0)

        mv.y = 0;

    if (x + (mv.x >> 1) < 0)

        mv.x = 0;



    src = &previous[(x + (mv.x >> 1)) + (y + (mv.y >> 1)) * pitch];

    dst = current;



    s->dsp.put_pixels_tab[0][(mv.y & 1) << 1 | (mv.x & 1)](dst, src, pitch, 16);



    return 0;

}
