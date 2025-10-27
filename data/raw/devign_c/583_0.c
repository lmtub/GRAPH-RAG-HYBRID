yuv2rgba64_2_c_template(SwsContext *c, const int32_t *buf[2],

                       const int32_t *ubuf[2], const int32_t *vbuf[2],

                       const int32_t *abuf[2], uint16_t *dest, int dstW,

                       int yalpha, int uvalpha, int y,

                       enum AVPixelFormat target, int hasAlpha, int eightbytes)

{

    const int32_t *buf0  = buf[0],  *buf1  = buf[1],

                  *ubuf0 = ubuf[0], *ubuf1 = ubuf[1],

                  *vbuf0 = vbuf[0], *vbuf1 = vbuf[1],

                  *abuf0 = hasAlpha ? abuf[0] : NULL,

                  *abuf1 = hasAlpha ? abuf[1] : NULL;

    int  yalpha1 = 4096 - yalpha;

    int uvalpha1 = 4096 - uvalpha;

    int i;

    int A1 = 0xffff<<14, A2 = 0xffff<<14;



    for (i = 0; i < ((dstW + 1) >> 1); i++) {

        int Y1 = (buf0[i * 2]     * yalpha1  + buf1[i * 2]     * yalpha) >> 14;

        int Y2 = (buf0[i * 2 + 1] * yalpha1  + buf1[i * 2 + 1] * yalpha) >> 14;

        int U  = (ubuf0[i]        * uvalpha1 + ubuf1[i]        * uvalpha + (-128 << 23)) >> 14;

        int V  = (vbuf0[i]        * uvalpha1 + vbuf1[i]        * uvalpha + (-128 << 23)) >> 14;

        int R, G, B;



        Y1 -= c->yuv2rgb_y_offset;

        Y2 -= c->yuv2rgb_y_offset;

        Y1 *= c->yuv2rgb_y_coeff;

        Y2 *= c->yuv2rgb_y_coeff;

        Y1 += 1 << 13;

        Y2 += 1 << 13;



        R = V * c->yuv2rgb_v2r_coeff;

        G = V * c->yuv2rgb_v2g_coeff + U * c->yuv2rgb_u2g_coeff;

        B =                            U * c->yuv2rgb_u2b_coeff;



        if (hasAlpha) {

            A1 = (abuf0[i * 2    ] * yalpha1 + abuf1[i * 2    ] * yalpha) >> 1;

            A2 = (abuf0[i * 2 + 1] * yalpha1 + abuf1[i * 2 + 1] * yalpha) >> 1;



            A1 += 1 << 13;

            A2 += 1 << 13;

        }



        output_pixel(&dest[0], av_clip_uintp2(R_B + Y1, 30) >> 14);

        output_pixel(&dest[1], av_clip_uintp2(  G + Y1, 30) >> 14);

        output_pixel(&dest[2], av_clip_uintp2(B_R + Y1, 30) >> 14);

        if (eightbytes) {

            output_pixel(&dest[3], av_clip_uintp2(A1      , 30) >> 14);

            output_pixel(&dest[4], av_clip_uintp2(R_B + Y2, 30) >> 14);

            output_pixel(&dest[5], av_clip_uintp2(  G + Y2, 30) >> 14);

            output_pixel(&dest[6], av_clip_uintp2(B_R + Y2, 30) >> 14);

            output_pixel(&dest[7], av_clip_uintp2(A2      , 30) >> 14);

            dest += 8;

        } else {

            output_pixel(&dest[3], av_clip_uintp2(R_B + Y2, 30) >> 14);

            output_pixel(&dest[4], av_clip_uintp2(  G + Y2, 30) >> 14);

            output_pixel(&dest[5], av_clip_uintp2(B_R + Y2, 30) >> 14);

            dest += 6;

        }

    }

}
