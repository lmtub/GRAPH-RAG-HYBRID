yuv2422_1_c_template(SwsContext *c, const int16_t *buf0,

                     const int16_t *ubuf[2], const int16_t *vbuf[2],

                     const int16_t *abuf0, uint8_t *dest, int dstW,

                     int uvalpha, int y, enum PixelFormat target)

{

    const int16_t *ubuf0 = ubuf[0], *ubuf1 = ubuf[1],

                  *vbuf0 = vbuf[0], *vbuf1 = vbuf[1];

    int i;



    if (uvalpha < 2048) {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 = buf0[i * 2]     >> 7;

            int Y2 = buf0[i * 2 + 1] >> 7;

            int U  = ubuf1[i]        >> 7;

            int V  = vbuf1[i]        >> 7;



            output_pixels(i * 4, Y1, U, Y2, V);

        }

    } else {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 =  buf0[i * 2]          >> 7;

            int Y2 =  buf0[i * 2 + 1]      >> 7;

            int U  = (ubuf0[i] + ubuf1[i]) >> 8;

            int V  = (vbuf0[i] + vbuf1[i]) >> 8;



            output_pixels(i * 4, Y1, U, Y2, V);

        }

    }

}
