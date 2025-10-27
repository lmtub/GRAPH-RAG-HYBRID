static int read_decoding_params(MLPDecodeContext *m, GetBitContext *gbp,

                                unsigned int substr)

{

    SubStream *s = &m->substream[substr];

    unsigned int ch;

    int ret;



    if (s->param_presence_flags & PARAM_PRESENCE)

        if (get_bits1(gbp))

            s->param_presence_flags = get_bits(gbp, 8);



    if (s->param_presence_flags & PARAM_BLOCKSIZE)

        if (get_bits1(gbp)) {

            s->blocksize = get_bits(gbp, 9);

            if (s->blocksize < 8 || s->blocksize > m->access_unit_size) {

                av_log(m->avctx, AV_LOG_ERROR, "Invalid blocksize.\n");

                s->blocksize = 0;

                return AVERROR_INVALIDDATA;

            }

        }



    if (s->param_presence_flags & PARAM_MATRIX)

        if (get_bits1(gbp))

            if ((ret = read_matrix_params(m, substr, gbp)) < 0)

                return ret;



    if (s->param_presence_flags & PARAM_OUTSHIFT)

        if (get_bits1(gbp)) {

            for (ch = 0; ch <= s->max_matrix_channel; ch++) {

                s->output_shift[ch] = get_sbits(gbp, 4);

                if (s->output_shift[ch] < 0) {

                    avpriv_request_sample(m->avctx, "Negative output_shift");

                    s->output_shift[ch] = 0;

                }

            }

            if (substr == m->max_decoded_substream)

                m->dsp.mlp_pack_output = m->dsp.mlp_select_pack_output(s->ch_assign,

                                                                       s->output_shift,

                                                                       s->max_matrix_channel,

                                                                       m->avctx->sample_fmt == AV_SAMPLE_FMT_S32);

        }



    if (s->param_presence_flags & PARAM_QUANTSTEP)

        if (get_bits1(gbp))

            for (ch = 0; ch <= s->max_channel; ch++) {

                ChannelParams *cp = &s->channel_params[ch];



                s->quant_step_size[ch] = get_bits(gbp, 4);



                cp->sign_huff_offset = calculate_sign_huff(m, substr, ch);

            }



    for (ch = s->min_channel; ch <= s->max_channel; ch++)

        if (get_bits1(gbp))

            if ((ret = read_channel_params(m, substr, gbp, ch)) < 0)

                return ret;



    return 0;

}
