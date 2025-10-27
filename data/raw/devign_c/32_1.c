static int huffman_decode(MPADecodeContext *s, GranuleDef *g,

                          int16_t *exponents, int end_pos2)

{

    int s_index;

    int i;

    int last_pos, bits_left;

    VLC *vlc;

    int end_pos = FFMIN(end_pos2, s->gb.size_in_bits);



    /* low frequencies (called big values) */

    s_index = 0;

    for (i = 0; i < 3; i++) {

        int j, k, l, linbits;

        j = g->region_size[i];

        if (j == 0)

            continue;

        /* select vlc table */

        k       = g->table_select[i];

        l       = mpa_huff_data[k][0];

        linbits = mpa_huff_data[k][1];

        vlc     = &huff_vlc[l];



        if (!l) {

            memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid) * 2 * j);

            s_index += 2 * j;

            continue;

        }



        /* read huffcode and compute each couple */

        for (; j > 0; j--) {

            int exponent, x, y;

            int v;

            int pos = get_bits_count(&s->gb);



            if (pos >= end_pos){

                switch_buffer(s, &pos, &end_pos, &end_pos2);

                if (pos >= end_pos)

                    break;

            }

            y = get_vlc2(&s->gb, vlc->table, 7, 3);



            if (!y) {

                g->sb_hybrid[s_index  ] =

                g->sb_hybrid[s_index+1] = 0;

                s_index += 2;

                continue;

            }



            exponent= exponents[s_index];



            ff_dlog(s->avctx, "region=%d n=%d x=%d y=%d exp=%d\n",

                    i, g->region_size[i] - j, x, y, exponent);

            if (y & 16) {

                x = y >> 5;

                y = y & 0x0f;

                if (x < 15) {

                    READ_FLIP_SIGN(g->sb_hybrid + s_index, RENAME(expval_table)[exponent] + x)

                } else {

                    x += get_bitsz(&s->gb, linbits);

                    v  = l3_unscale(x, exponent);

                    if (get_bits1(&s->gb))

                        v = -v;

                    g->sb_hybrid[s_index] = v;

                }

                if (y < 15) {

                    READ_FLIP_SIGN(g->sb_hybrid + s_index + 1, RENAME(expval_table)[exponent] + y)

                } else {

                    y += get_bitsz(&s->gb, linbits);

                    v  = l3_unscale(y, exponent);

                    if (get_bits1(&s->gb))

                        v = -v;

                    g->sb_hybrid[s_index+1] = v;

                }

            } else {

                x = y >> 5;

                y = y & 0x0f;

                x += y;

                if (x < 15) {

                    READ_FLIP_SIGN(g->sb_hybrid + s_index + !!y, RENAME(expval_table)[exponent] + x)

                } else {

                    x += get_bitsz(&s->gb, linbits);

                    v  = l3_unscale(x, exponent);

                    if (get_bits1(&s->gb))

                        v = -v;

                    g->sb_hybrid[s_index+!!y] = v;

                }

                g->sb_hybrid[s_index + !y] = 0;

            }

            s_index += 2;

        }

    }



    /* high frequencies */

    vlc = &huff_quad_vlc[g->count1table_select];

    last_pos = 0;

    while (s_index <= 572) {

        int pos, code;

        pos = get_bits_count(&s->gb);

        if (pos >= end_pos) {

            if (pos > end_pos2 && last_pos) {

                /* some encoders generate an incorrect size for this

                   part. We must go back into the data */

                s_index -= 4;

                skip_bits_long(&s->gb, last_pos - pos);

                av_log(s->avctx, AV_LOG_INFO, "overread, skip %d enddists: %d %d\n", last_pos - pos, end_pos-pos, end_pos2-pos);

                if(s->err_recognition & AV_EF_BITSTREAM)

                    s_index=0;

                break;

            }

            switch_buffer(s, &pos, &end_pos, &end_pos2);

            if (pos >= end_pos)

                break;

        }

        last_pos = pos;



        code = get_vlc2(&s->gb, vlc->table, vlc->bits, 1);

        ff_dlog(s->avctx, "t=%d code=%d\n", g->count1table_select, code);

        g->sb_hybrid[s_index+0] =

        g->sb_hybrid[s_index+1] =

        g->sb_hybrid[s_index+2] =

        g->sb_hybrid[s_index+3] = 0;

        while (code) {

            static const int idxtab[16] = { 3,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0 };

            int v;

            int pos = s_index + idxtab[code];

            code   ^= 8 >> idxtab[code];

            READ_FLIP_SIGN(g->sb_hybrid + pos, RENAME(exp_table)+exponents[pos])

        }

        s_index += 4;

    }

    /* skip extension bits */

    bits_left = end_pos2 - get_bits_count(&s->gb);

    if (bits_left < 0 && (s->err_recognition & AV_EF_BUFFER)) {

        av_log(s->avctx, AV_LOG_ERROR, "bits_left=%d\n", bits_left);

        s_index=0;

    } else if (bits_left > 0 && (s->err_recognition & AV_EF_BUFFER)) {

        av_log(s->avctx, AV_LOG_ERROR, "bits_left=%d\n", bits_left);

        s_index = 0;

    }

    memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid) * (576 - s_index));

    skip_bits_long(&s->gb, bits_left);



    i = get_bits_count(&s->gb);

    switch_buffer(s, &i, &end_pos, &end_pos2);



    return 0;

}
