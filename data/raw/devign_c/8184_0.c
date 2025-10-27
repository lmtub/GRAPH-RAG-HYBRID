static int tscc2_decode_mb(TSCC2Context *c, int *q, int vlc_set,

                           uint8_t *dst, int stride, int plane)

{

    GetBitContext *gb = &c->gb;

    int prev_dc, dc, nc, ac, bpos, val;

    int i, j, k, l;



    if (get_bits1(gb)) {

        if (get_bits1(gb)) {

            val = get_bits(gb, 8);

            for (i = 0; i < 8; i++, dst += stride)

                memset(dst, val, 16);

        } else {

            if (get_bits_left(gb) < 16 * 8 * 8)

                return AVERROR_INVALIDDATA;

            for (i = 0; i < 8; i++) {

                for (j = 0; j < 16; j++)

                    dst[j] = get_bits(gb, 8);

                dst += stride;

            }

        }

        return 0;

    }



    prev_dc = 0;

    for (j = 0; j < 2; j++) {

        for (k = 0; k < 4; k++) {

            if (!(j | k)) {

                dc = get_bits(gb, 8);

            } else {

                dc = get_vlc2(gb, c->dc_vlc.table, 9, 2);

                if (dc == -1)

                    return AVERROR_INVALIDDATA;

                if (dc == 0x100)

                    dc = get_bits(gb, 8);

            }

            dc          = (dc + prev_dc) & 0xFF;

            prev_dc     = dc;

            c->block[0] = dc;



            nc = get_vlc2(gb, c->nc_vlc[vlc_set].table, 9, 1);

            if (nc == -1)

                return AVERROR_INVALIDDATA;



            bpos = 1;

            memset(c->block + 1, 0, 15 * sizeof(*c->block));

            for (l = 0; l < nc; l++) {

                ac = get_vlc2(gb, c->ac_vlc[vlc_set].table, 9, 2);

                if (ac == -1)

                    return AVERROR_INVALIDDATA;

                if (ac == 0x1000)

                    ac = get_bits(gb, 12);

                bpos += ac & 0xF;

                if (bpos >= 64)

                    return AVERROR_INVALIDDATA;

                val = sign_extend(ac >> 4, 8);

                c->block[tscc2_zigzag[bpos++]] = val;

            }

            tscc2_idct4_put(c->block, q, dst + k * 4, stride);

        }

        dst += 4 * stride;

    }

    return 0;

}
