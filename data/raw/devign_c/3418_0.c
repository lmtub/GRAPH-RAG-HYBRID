yuv2mono_2_c_template(SwsContext *c, const int16_t *buf[2],

                      const int16_t *ubuf[2], const int16_t *vbuf[2],

                      const int16_t *abuf[2], uint8_t *dest, int dstW,

                      int yalpha, int uvalpha, int y,

                      enum AVPixelFormat target)

{

    const int16_t *buf0  = buf[0],  *buf1  = buf[1];

    const uint8_t * const d128 = dither_8x8_220[y & 7];

    int  yalpha1 = 4096 - yalpha;

    int i;



    if (c->flags & SWS_ERROR_DIFFUSION) {

        int err = 0;

        int acc = 0;

        for (i = 0; i < dstW; i +=2) {

            int Y;



            Y = (buf0[i + 0] * yalpha1 + buf1[i + 0] * yalpha) >> 19;

            Y += (7*err + 1*c->dither_error[0][i] + 5*c->dither_error[0][i+1] + 3*c->dither_error[0][i+2] + 8 - 256)>>4;

            c->dither_error[0][i] = err;

            acc = 2*acc + (Y >= 128);

            Y -= 220*(acc&1);



            err = (buf0[i + 1] * yalpha1 + buf1[i + 1] * yalpha) >> 19;

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

        int Y, acc = 0;



        Y = (buf0[i + 0] * yalpha1 + buf1[i + 0] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[0]);

        Y = (buf0[i + 1] * yalpha1 + buf1[i + 1] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[1]);

        Y = (buf0[i + 2] * yalpha1 + buf1[i + 2] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[2]);

        Y = (buf0[i + 3] * yalpha1 + buf1[i + 3] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[3]);

        Y = (buf0[i + 4] * yalpha1 + buf1[i + 4] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[4]);

        Y = (buf0[i + 5] * yalpha1 + buf1[i + 5] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[5]);

        Y = (buf0[i + 6] * yalpha1 + buf1[i + 6] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[6]);

        Y = (buf0[i + 7] * yalpha1 + buf1[i + 7] * yalpha) >> 19;

        accumulate_bit(acc, Y + d128[7]);



        output_pixel(*dest++, acc);

    }

    }

}
