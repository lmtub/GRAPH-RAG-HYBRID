yuv2rgb_full_2_c_template(SwsContext *c, const int16_t *buf[2],

                     const int16_t *ubuf[2], const int16_t *vbuf[2],

                     const int16_t *abuf[2], uint8_t *dest, int dstW,

                     int yalpha, int uvalpha, int y,

                     enum AVPixelFormat target, int hasAlpha)

{

    const int16_t *buf0  = buf[0],  *buf1  = buf[1],

                  *ubuf0 = ubuf[0], *ubuf1 = ubuf[1],

                  *vbuf0 = vbuf[0], *vbuf1 = vbuf[1],

                  *abuf0 = hasAlpha ? abuf[0] : NULL,

                  *abuf1 = hasAlpha ? abuf[1] : NULL;

    int  yalpha1 = 4096 - yalpha;

    int uvalpha1 = 4096 - uvalpha;

    int i;

    int step = (target == AV_PIX_FMT_RGB24 || target == AV_PIX_FMT_BGR24) ? 3 : 4;

    int err[4] = {0};



    if(   target == AV_PIX_FMT_BGR4_BYTE || target == AV_PIX_FMT_RGB4_BYTE

       || target == AV_PIX_FMT_BGR8      || target == AV_PIX_FMT_RGB8)

        step = 1;



    for (i = 0; i < dstW; i++) {

        int Y = ( buf0[i] * yalpha1  +  buf1[i] * yalpha             ) >> 10; //FIXME rounding

        int U = (ubuf0[i] * uvalpha1 + ubuf1[i] * uvalpha-(128 << 19)) >> 10;

        int V = (vbuf0[i] * uvalpha1 + vbuf1[i] * uvalpha-(128 << 19)) >> 10;

        int A;



        if (hasAlpha) {

            A = (abuf0[i] * yalpha1 + abuf1[i] * yalpha + (1<<18)) >> 19;

            if (A & 0x100)

                A = av_clip_uint8(A);

        }



        yuv2rgb_write_full(c, dest, i, Y, A, U, V, y, target, hasAlpha, err);

        dest += step;

    }

    c->dither_error[0][i] = err[0];

    c->dither_error[1][i] = err[1];

    c->dither_error[2][i] = err[2];

}
