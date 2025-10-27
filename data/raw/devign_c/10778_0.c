yuv2rgb48_1_c_template(SwsContext *c, const uint16_t *buf0,

                       const uint16_t *ubuf0, const uint16_t *ubuf1,

                       const uint16_t *vbuf0, const uint16_t *vbuf1,

                       const uint16_t *abuf0, uint8_t *dest, int dstW,

                       int uvalpha, enum PixelFormat dstFormat,

                       int flags, int y, enum PixelFormat target)

{

    int i;



    if (uvalpha < 2048) {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 = buf0[i * 2]     >> 7;

            int Y2 = buf0[i * 2 + 1] >> 7;

            int U  = ubuf1[i]        >> 7;

            int V  = vbuf1[i]        >> 7;

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

    } else {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 =  buf0[i * 2]          >> 7;

            int Y2 =  buf0[i * 2 + 1]      >> 7;

            int U  = (ubuf0[i] + ubuf1[i]) >> 8;

            int V  = (vbuf0[i] + vbuf1[i]) >> 8;

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

}
