yuv2rgb_full_X_c_template(SwsContext *c, const int16_t *lumFilter,

                          const int16_t **lumSrc, int lumFilterSize,

                          const int16_t *chrFilter, const int16_t **chrUSrc,

                          const int16_t **chrVSrc, int chrFilterSize,

                          const int16_t **alpSrc, uint8_t *dest,

                          int dstW, int y, enum AVPixelFormat target, int hasAlpha)

{

    int i;

    int step = (target == AV_PIX_FMT_RGB24 || target == AV_PIX_FMT_BGR24) ? 3 : 4;

    int err[4] = {0};



    if(   target == AV_PIX_FMT_BGR4_BYTE || target == AV_PIX_FMT_RGB4_BYTE

       || target == AV_PIX_FMT_BGR8      || target == AV_PIX_FMT_RGB8)

        step = 1;



    for (i = 0; i < dstW; i++) {

        int j;

        int Y = 1<<9;

        int U = (1<<9)-(128 << 19);

        int V = (1<<9)-(128 << 19);

        int A;



        for (j = 0; j < lumFilterSize; j++) {

            Y += lumSrc[j][i] * lumFilter[j];

        }

        for (j = 0; j < chrFilterSize; j++) {

            U += chrUSrc[j][i] * chrFilter[j];

            V += chrVSrc[j][i] * chrFilter[j];

        }

        Y >>= 10;

        U >>= 10;

        V >>= 10;

        if (hasAlpha) {

            A = 1 << 18;

            for (j = 0; j < lumFilterSize; j++) {

                A += alpSrc[j][i] * lumFilter[j];

            }

            A >>= 19;

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
