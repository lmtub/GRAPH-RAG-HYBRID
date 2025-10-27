static int decode_coeffs(WMAProDecodeCtx *s, int c)

{

    /* Integers 0..15 as single-precision floats.  The table saves a

       costly int to float conversion, and storing the values as

       integers allows fast sign-flipping. */

    static const int fval_tab[16] = {

        0x00000000, 0x3f800000, 0x40000000, 0x40400000,

        0x40800000, 0x40a00000, 0x40c00000, 0x40e00000,

        0x41000000, 0x41100000, 0x41200000, 0x41300000,

        0x41400000, 0x41500000, 0x41600000, 0x41700000,

    };

    int vlctable;

    VLC* vlc;

    WMAProChannelCtx* ci = &s->channel[c];

    int rl_mode = 0;

    int cur_coeff = 0;

    int num_zeros = 0;

    const uint16_t* run;

    const float* level;



    av_dlog(s->avctx, "decode coefficients for channel %i\n", c);



    vlctable = get_bits1(&s->gb);

    vlc = &coef_vlc[vlctable];



    if (vlctable) {

        run = coef1_run;

        level = coef1_level;

    } else {

        run = coef0_run;

        level = coef0_level;

    }



    /** decode vector coefficients (consumes up to 167 bits per iteration for

      4 vector coded large values) */

    while ((s->transmit_num_vec_coeffs || !rl_mode) &&

           (cur_coeff + 3 < ci->num_vec_coeffs)) {

        int vals[4];

        int i;

        unsigned int idx;



        idx = get_vlc2(&s->gb, vec4_vlc.table, VLCBITS, VEC4MAXDEPTH);



        if (idx == HUFF_VEC4_SIZE - 1) {

            for (i = 0; i < 4; i += 2) {

                idx = get_vlc2(&s->gb, vec2_vlc.table, VLCBITS, VEC2MAXDEPTH);

                if (idx == HUFF_VEC2_SIZE - 1) {

                    int v0, v1;

                    v0 = get_vlc2(&s->gb, vec1_vlc.table, VLCBITS, VEC1MAXDEPTH);

                    if (v0 == HUFF_VEC1_SIZE - 1)

                        v0 += ff_wma_get_large_val(&s->gb);

                    v1 = get_vlc2(&s->gb, vec1_vlc.table, VLCBITS, VEC1MAXDEPTH);

                    if (v1 == HUFF_VEC1_SIZE - 1)

                        v1 += ff_wma_get_large_val(&s->gb);

                    ((float*)vals)[i  ] = v0;

                    ((float*)vals)[i+1] = v1;

                } else {

                    vals[i]   = fval_tab[symbol_to_vec2[idx] >> 4 ];

                    vals[i+1] = fval_tab[symbol_to_vec2[idx] & 0xF];

                }

            }

        } else {

            vals[0] = fval_tab[ symbol_to_vec4[idx] >> 12      ];

            vals[1] = fval_tab[(symbol_to_vec4[idx] >> 8) & 0xF];

            vals[2] = fval_tab[(symbol_to_vec4[idx] >> 4) & 0xF];

            vals[3] = fval_tab[ symbol_to_vec4[idx]       & 0xF];

        }



        /** decode sign */

        for (i = 0; i < 4; i++) {

            if (vals[i]) {

                int sign = get_bits1(&s->gb) - 1;

                *(uint32_t*)&ci->coeffs[cur_coeff] = vals[i] ^ sign<<31;

                num_zeros = 0;

            } else {

                ci->coeffs[cur_coeff] = 0;

                /** switch to run level mode when subframe_len / 128 zeros

                    were found in a row */

                rl_mode |= (++num_zeros > s->subframe_len >> 8);

            }

            ++cur_coeff;

        }

    }



    /** decode run level coded coefficients */

    if (cur_coeff < s->subframe_len) {

        memset(&ci->coeffs[cur_coeff], 0,

               sizeof(*ci->coeffs) * (s->subframe_len - cur_coeff));

        if (ff_wma_run_level_decode(s->avctx, &s->gb, vlc,

                                    level, run, 1, ci->coeffs,

                                    cur_coeff, s->subframe_len,

                                    s->subframe_len, s->esc_len, 0))

            return AVERROR_INVALIDDATA;

    }



    return 0;

}
