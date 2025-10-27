static inline int read_huff_channels(MLPDecodeContext *m, GetBitContext *gbp,

                                     unsigned int substr, unsigned int pos)

{

    SubStream *s = &m->substream[substr];

    unsigned int mat, channel;



    for (mat = 0; mat < s->num_primitive_matrices; mat++)

        if (s->lsb_bypass[mat])

            m->bypassed_lsbs[pos + s->blockpos][mat] = get_bits1(gbp);



    for (channel = s->min_channel; channel <= s->max_channel; channel++) {

        ChannelParams *cp = &s->channel_params[channel];

        int codebook = cp->codebook;

        int quant_step_size = s->quant_step_size[channel];

        int lsb_bits = cp->huff_lsbs - quant_step_size;

        int result = 0;



        if (codebook > 0)

            result = get_vlc2(gbp, huff_vlc[codebook-1].table,

                            VLC_BITS, (9 + VLC_BITS - 1) / VLC_BITS);



        if (result < 0)

            return AVERROR_INVALIDDATA;



        if (lsb_bits > 0)

            result = (result << lsb_bits) + get_bits(gbp, lsb_bits);



        result  += cp->sign_huff_offset;

        result <<= quant_step_size;



        m->sample_buffer[pos + s->blockpos][channel] = result;

    }



    return 0;

}
