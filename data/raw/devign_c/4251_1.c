yuv2rgb_full_1_c_template(SwsContext *c, const int16_t *buf0,

                     const int16_t *ubuf[2], const int16_t *vbuf[2],

                     const int16_t *abuf0, uint8_t *dest, int dstW,

                     int uvalpha, int y, enum AVPixelFormat target,

                     int hasAlpha)

{

    const int16_t *ubuf0 = ubuf[0], *vbuf0 = vbuf[0];

    int i;

    int step = (target == AV_PIX_FMT_RGB24 || target == AV_PIX_FMT_BGR24) ? 3 : 4;

    int err[4] = {0};



    if(   target == AV_PIX_FMT_BGR4_BYTE || target == AV_PIX_FMT_RGB4_BYTE

       || target == AV_PIX_FMT_BGR8      || target == AV_PIX_FMT_RGB8)

        step = 1;



    if (uvalpha < 2048) {

        for (i = 0; i < dstW; i++) {

            int Y = buf0[i] << 2;

            int U = (ubuf0[i] - (128<<7)) << 2;

            int V = (vbuf0[i] - (128<<7)) << 2;

            int A;



            if (hasAlpha) {

                A = (abuf0[i] + 64) >> 7;

                if (A & 0x100)

                    A = av_clip_uint8(A);

            }



            yuv2rgb_write_full(c, dest, i, Y, A, U, V, y, target, hasAlpha, err);

            dest += step;

        }

    } else {

        const int16_t *ubuf1 = ubuf[1], *vbuf1 = vbuf[1];

        for (i = 0; i < dstW; i++) {

            int Y = buf0[i] << 2;

            int U = (ubuf0[i] + ubuf1[i] - (128<<8)) << 1;

            int V = (vbuf0[i] + vbuf1[i] - (128<<8)) << 1;

            int A;



            if (hasAlpha) {

                A = (abuf0[i] + 64) >> 7;

                if (A & 0x100)

                    A = av_clip_uint8(A);

            }



            yuv2rgb_write_full(c, dest, i, Y, A, U, V, y, target, hasAlpha, err);

            dest += step;

        }

    }



    c->dither_error[0][i] = err[0];

    c->dither_error[1][i] = err[1];

    c->dither_error[2][i] = err[2];

}
