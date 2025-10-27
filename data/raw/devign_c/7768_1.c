yuv2rgb48_1_c_template(SwsContext *c, const int32_t *buf0,

                       const int32_t *ubuf[2], const int32_t *vbuf[2],

                       const int32_t *abuf0, uint16_t *dest, int dstW,

                       int uvalpha, int y, enum PixelFormat target)

{

    const int32_t *ubuf0 = ubuf[0], *ubuf1 = ubuf[1],

                  *vbuf0 = vbuf[0], *vbuf1 = vbuf[1];

    int i;



    if (uvalpha < 2048) {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 = (buf0[i * 2]    ) >> 2;

            int Y2 = (buf0[i * 2 + 1]) >> 2;

            int U  = (ubuf0[i] + (-128 << 11)) >> 2;

            int V  = (vbuf0[i] + (-128 << 11)) >> 2;

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



            output_pixel(&dest[0], av_clip_uintp2(R_B + Y1, 30) >> 14);

            output_pixel(&dest[1], av_clip_uintp2(  G + Y1, 30) >> 14);

            output_pixel(&dest[2], av_clip_uintp2(B_R + Y1, 30) >> 14);

            output_pixel(&dest[3], av_clip_uintp2(R_B + Y2, 30) >> 14);

            output_pixel(&dest[4], av_clip_uintp2(  G + Y2, 30) >> 14);

            output_pixel(&dest[5], av_clip_uintp2(B_R + Y2, 30) >> 14);

            dest += 6;

        }

    } else {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 = (buf0[i * 2]    ) >> 2;

            int Y2 = (buf0[i * 2 + 1]) >> 2;

            int U  = (ubuf0[i] + ubuf1[i] + (-128 << 11)) >> 3;

            int V  = (vbuf0[i] + vbuf1[i] + (-128 << 11)) >> 3;

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



            output_pixel(&dest[0], av_clip_uintp2(R_B + Y1, 30) >> 14);

            output_pixel(&dest[1], av_clip_uintp2(  G + Y1, 30) >> 14);

            output_pixel(&dest[2], av_clip_uintp2(B_R + Y1, 30) >> 14);

            output_pixel(&dest[3], av_clip_uintp2(R_B + Y2, 30) >> 14);

            output_pixel(&dest[4], av_clip_uintp2(  G + Y2, 30) >> 14);

            output_pixel(&dest[5], av_clip_uintp2(B_R + Y2, 30) >> 14);

            dest += 6;

        }

    }

}
