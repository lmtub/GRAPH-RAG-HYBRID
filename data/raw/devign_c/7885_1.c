yuv2rgb_2_c_template(SwsContext *c, const int16_t *buf[2],

                     const int16_t *ubuf[2], const int16_t *vbuf[2],

                     const int16_t *abuf[2], uint8_t *dest, int dstW,

                     int yalpha, int uvalpha, int y,

                     enum PixelFormat target, int hasAlpha)

{

    const int16_t *buf0  = buf[0],  *buf1  = buf[1],

                  *ubuf0 = ubuf[0], *ubuf1 = ubuf[1],

                  *vbuf0 = vbuf[0], *vbuf1 = vbuf[1],

                  *abuf0 = abuf[0], *abuf1 = abuf[1];

    int  yalpha1 = 4095 - yalpha;

    int uvalpha1 = 4095 - uvalpha;

    int i;



    for (i = 0; i < (dstW >> 1); i++) {

        int Y1 = (buf0[i * 2]     * yalpha1  + buf1[i * 2]     * yalpha)  >> 19;

        int Y2 = (buf0[i * 2 + 1] * yalpha1  + buf1[i * 2 + 1] * yalpha)  >> 19;

        int U  = (ubuf0[i]        * uvalpha1 + ubuf1[i]        * uvalpha) >> 19;

        int V  = (vbuf0[i]        * uvalpha1 + vbuf1[i]        * uvalpha) >> 19;

        int A1, A2;

        const void *r =  c->table_rV[V],

                   *g = (c->table_gU[U] + c->table_gV[V]),

                   *b =  c->table_bU[U];



        if (hasAlpha) {

            A1 = (abuf0[i * 2    ] * yalpha1 + abuf1[i * 2    ] * yalpha) >> 19;

            A2 = (abuf0[i * 2 + 1] * yalpha1 + abuf1[i * 2 + 1] * yalpha) >> 19;

        }



        yuv2rgb_write(dest, i, Y1, Y2, U, V, hasAlpha ? A1 : 0, hasAlpha ? A2 : 0,

                      r, g, b, y, target, hasAlpha);

    }

}
