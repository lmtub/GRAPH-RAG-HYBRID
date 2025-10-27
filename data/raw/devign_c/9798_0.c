yuv2rgb48_2_c_template(SwsContext *c, const uint16_t *buf0,

                       const uint16_t *buf1, const uint16_t *ubuf0,

                       const uint16_t *ubuf1, const uint16_t *vbuf0,

                       const uint16_t *vbuf1, const uint16_t *abuf0,

                       const uint16_t *abuf1, uint8_t *dest, int dstW,

                       int yalpha, int uvalpha, int y,

                       enum PixelFormat target)

{

    int  yalpha1 = 4095 - yalpha;

    int uvalpha1 = 4095 - uvalpha;

    int i;



    for (i = 0; i < (dstW >> 1); i++) {

        int Y1 = (buf0[i * 2]     * yalpha1  + buf1[i * 2]     * yalpha)  >> 19;

        int Y2 = (buf0[i * 2 + 1] * yalpha1  + buf1[i * 2 + 1] * yalpha)  >> 19;

        int U  = (ubuf0[i]        * uvalpha1 + ubuf1[i]        * uvalpha) >> 19;

        int V  = (vbuf0[i]        * uvalpha1 + vbuf1[i]        * uvalpha) >> 19;

        const uint8_t *r = (const uint8_t *) c->table_rV[V],

                      *g = (const uint8_t *)(c->table_gU[U] + c->table_gV[V]),

                      *b = (const uint8_t *) c->table_bU[U];



        dest[ 0] = dest[ 1] = r_b[Y1];

        dest[ 2] = dest[ 3] =   g[Y1];

        dest[ 4] = dest[ 5] = b_r[Y1];

        dest[ 6] = dest[ 7] = r_b[Y2];

        dest[ 8] = dest[ 9] =   g[Y2];

        dest[10] = dest[11] = b_r[Y2];

        dest += 12;

    }

}
