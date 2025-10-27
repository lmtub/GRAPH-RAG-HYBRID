yuv2rgba64_full_2_c_template(SwsContext *c, const int32_t *buf[2],

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

    int A = 0xffff<<14;



    for (i = 0; i < dstW; i++) {

        int Y  = (buf0[i]     * yalpha1  + buf1[i]     * yalpha) >> 14;

        int U  = (ubuf0[i]   * uvalpha1 + ubuf1[i]     * uvalpha + (-128 << 23)) >> 14;

        int V  = (vbuf0[i]   * uvalpha1 + vbuf1[i]     * uvalpha + (-128 << 23)) >> 14;

        int R, G, B;



        Y -= c->yuv2rgb_y_offset;

        Y *= c->yuv2rgb_y_coeff;

        Y += 1 << 13;



        R = V * c->yuv2rgb_v2r_coeff;

        G = V * c->yuv2rgb_v2g_coeff + U * c->yuv2rgb_u2g_coeff;

        B =                            U * c->yuv2rgb_u2b_coeff;



        if (hasAlpha) {

            A = (abuf0[i] * yalpha1 + abuf1[i] * yalpha) >> 1;



            A += 1 << 13;

        }



        output_pixel(&dest[0], av_clip_uintp2(R_B + Y, 30) >> 14);

        output_pixel(&dest[1], av_clip_uintp2(  G + Y, 30) >> 14);

        output_pixel(&dest[2], av_clip_uintp2(B_R + Y, 30) >> 14);

        if (eightbytes) {

            output_pixel(&dest[3], av_clip_uintp2(A, 30) >> 14);

            dest += 4;

        } else {

            dest += 3;

        }

    }

}
