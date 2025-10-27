static int read_low_coeffs(AVCodecContext *avctx, int16_t *dst, int size, int width, ptrdiff_t stride)

{

    PixletContext *ctx = avctx->priv_data;

    GetBitContext *b = &ctx->gbit;

    unsigned cnt1, nbits, k, j = 0, i = 0;

    int64_t value, state = 3;

    int rlen, escape, flag = 0;



    while (i < size) {

        nbits = FFMIN(ff_clz((state >> 8) + 3) ^ 0x1F, 14);



        cnt1 = get_unary(b, 0, 8);

        if (cnt1 < 8) {

            value = show_bits(b, nbits);

            if (value <= 1) {

                skip_bits(b, nbits - 1);

                escape = ((1 << nbits) - 1) * cnt1;

            } else {

                skip_bits(b, nbits);

                escape = value + ((1 << nbits) - 1) * cnt1 - 1;

            }

        } else {

            escape = get_bits(b, 16);

        }



        value = -((escape + flag) & 1) | 1;

        dst[j++] = value * ((escape + flag + 1) >> 1);

        i++;

        if (j == width) {

            j = 0;

            dst += stride;

        }

        state = 120 * (escape + flag) + state - (120 * state >> 8);

        flag = 0;



        if (state * 4 > 0xFF || i >= size)

            continue;



        nbits = ((state + 8) >> 5) + (state ? ff_clz(state) : 32) - 24;

        escape = av_mod_uintp2(16383, nbits);

        cnt1 = get_unary(b, 0, 8);

        if (cnt1 > 7) {

            rlen = get_bits(b, 16);

        } else {

            value = show_bits(b, nbits);

            if (value > 1) {

                skip_bits(b, nbits);

                rlen = value + escape * cnt1 - 1;

            } else {

                skip_bits(b, nbits - 1);

                rlen = escape * cnt1;

            }

        }



        if (rlen > size - i)

            return AVERROR_INVALIDDATA;

        i += rlen;



        for (k = 0; k < rlen; k++) {

            dst[j++] = 0;

            if (j == width) {

                j = 0;

                dst += stride;

            }

        }



        state = 0;

        flag = rlen < 0xFFFF ? 1 : 0;

    }



    align_get_bits(b);

    return get_bits_count(b) >> 3;

}
