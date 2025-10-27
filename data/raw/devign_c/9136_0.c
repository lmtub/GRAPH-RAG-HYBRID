static void apply_unsharp(      uint8_t *dst, int dst_stride,

                          const uint8_t *src, int src_stride,

                          int width, int height, FilterParam *fp)

{

    uint32_t **sc = fp->sc;

    uint32_t sr[MAX_MATRIX_SIZE - 1], tmp1, tmp2;



    int32_t res;

    int x, y, z;

    const uint8_t *src2 = NULL;  //silence a warning



    if (!fp->amount) {

        if (dst_stride == src_stride)

            memcpy(dst, src, src_stride * height);

        else

            for (y = 0; y < height; y++, dst += dst_stride, src += src_stride)

                memcpy(dst, src, width);

        return;

    }



    for (y = 0; y < 2 * fp->steps_y; y++)

        memset(sc[y], 0, sizeof(sc[y][0]) * (width + 2 * fp->steps_x));



    for (y = -fp->steps_y; y < height + fp->steps_y; y++) {

        if (y < height)

            src2 = src;



        memset(sr, 0, sizeof(sr[0]) * (2 * fp->steps_x - 1));

        for (x = -fp->steps_x; x < width + fp->steps_x; x++) {

            tmp1 = x <= 0 ? src2[0] : x >= width ? src2[width-1] : src2[x];

            for (z = 0; z < fp->steps_x * 2; z += 2) {

                tmp2 = sr[z + 0] + tmp1; sr[z + 0] = tmp1;

                tmp1 = sr[z + 1] + tmp2; sr[z + 1] = tmp2;

            }

            for (z = 0; z < fp->steps_y * 2; z += 2) {

                tmp2 = sc[z + 0][x + fp->steps_x] + tmp1; sc[z + 0][x + fp->steps_x] = tmp1;

                tmp1 = sc[z + 1][x + fp->steps_x] + tmp2; sc[z + 1][x + fp->steps_x] = tmp2;

            }

            if (x >= fp->steps_x && y >= fp->steps_y) {

                const uint8_t *srx = src - fp->steps_y * src_stride + x - fp->steps_x;

                uint8_t *dsx       = dst - fp->steps_y * dst_stride + x - fp->steps_x;



                res = (int32_t)*srx + ((((int32_t) * srx - (int32_t)((tmp1 + fp->halfscale) >> fp->scalebits)) * fp->amount) >> 16);

                *dsx = av_clip_uint8(res);

            }

        }

        if (y >= 0) {

            dst += dst_stride;

            src += src_stride;

        }

    }

}
