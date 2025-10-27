yuv2mono_1_c_template(SwsContext *c, const int16_t *buf0,

                      const int16_t *ubuf[2], const int16_t *vbuf[2],

                      const int16_t *abuf0, uint8_t *dest, int dstW,

                      int uvalpha, int y, enum AVPixelFormat target)

{

    const uint8_t * const d128 = dither_8x8_220[y & 7];

    int i;



    if (c->flags & SWS_ERROR_DIFFUSION) {

        int err = 0;

        int acc = 0;

        for (i = 0; i < dstW; i +=2) {

            int Y;



            Y = ((buf0[i + 0] + 64) >> 7);

            Y += (7*err + 1*c->dither_error[0][i] + 5*c->dither_error[0][i+1] + 3*c->dither_error[0][i+2] + 8 - 256)>>4;

            c->dither_error[0][i] = err;

            acc = 2*acc + (Y >= 128);

            Y -= 220*(acc&1);



            err = ((buf0[i + 1] + 64) >> 7);

            err += (7*Y + 1*c->dither_error[0][i+1] + 5*c->dither_error[0][i+2] + 3*c->dither_error[0][i+3] + 8 - 256)>>4;

            c->dither_error[0][i+1] = Y;

            acc = 2*acc + (err >= 128);

            err -= 220*(acc&1);



            if ((i & 7) == 6)

                output_pixel(*dest++, acc);

        }

        c->dither_error[0][i] = err;

    } else {

    for (i = 0; i < dstW; i += 8) {

        int acc = 0;

        accumulate_bit(acc, ((buf0[i + 0] + 64) >> 7) + d128[0]);

        accumulate_bit(acc, ((buf0[i + 1] + 64) >> 7) + d128[1]);

        accumulate_bit(acc, ((buf0[i + 2] + 64) >> 7) + d128[2]);

        accumulate_bit(acc, ((buf0[i + 3] + 64) >> 7) + d128[3]);

        accumulate_bit(acc, ((buf0[i + 4] + 64) >> 7) + d128[4]);

        accumulate_bit(acc, ((buf0[i + 5] + 64) >> 7) + d128[5]);

        accumulate_bit(acc, ((buf0[i + 6] + 64) >> 7) + d128[6]);

        accumulate_bit(acc, ((buf0[i + 7] + 64) >> 7) + d128[7]);



        output_pixel(*dest++, acc);

    }

    }

}
