static int read_decoding_params(MLPDecodeContext *m, GetBitContext *gbp,

                                unsigned int substr)

{

    SubStream *s = &m->substream[substr];

    unsigned int ch;



    if (s->param_presence_flags & PARAM_PRESENCE)

    if (get_bits1(gbp))

        s->param_presence_flags = get_bits(gbp, 8);



    if (s->param_presence_flags & PARAM_BLOCKSIZE)

        if (get_bits1(gbp)) {

            s->blocksize = get_bits(gbp, 9);

            if (s->blocksize > MAX_BLOCKSIZE) {

                av_log(m->avctx, AV_LOG_ERROR, "block size too large\n");

                s->blocksize = 0;

                return -1;

            }

        }



    if (s->param_presence_flags & PARAM_MATRIX)

        if (get_bits1(gbp)) {

            if (read_matrix_params(m, s, gbp) < 0)

                return -1;

        }



    if (s->param_presence_flags & PARAM_OUTSHIFT)

        if (get_bits1(gbp))

            for (ch = 0; ch <= s->max_matrix_channel; ch++) {

                s->output_shift[ch] = get_sbits(gbp, 4);

                dprintf(m->avctx, "output shift[%d] = %d\n",

                        ch, s->output_shift[ch]);

            }



    if (s->param_presence_flags & PARAM_QUANTSTEP)

        if (get_bits1(gbp))

            for (ch = 0; ch <= s->max_channel; ch++) {

                ChannelParams *cp = &m->channel_params[ch];



                s->quant_step_size[ch] = get_bits(gbp, 4);



                cp->sign_huff_offset = calculate_sign_huff(m, substr, ch);

            }



    for (ch = s->min_channel; ch <= s->max_channel; ch++)

        if (get_bits1(gbp)) {

            if (read_channel_params(m, substr, gbp, ch) < 0)

                return -1;

        }



    return 0;

}
