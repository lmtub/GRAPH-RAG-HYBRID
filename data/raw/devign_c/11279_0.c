static int read_channel_params(MLPDecodeContext *m, unsigned int substr,

                               GetBitContext *gbp, unsigned int ch)

{

    ChannelParams *cp = &m->channel_params[ch];

    FilterParams *fir = &cp->filter_params[FIR];

    FilterParams *iir = &cp->filter_params[IIR];

    SubStream *s = &m->substream[substr];



    if (s->param_presence_flags & PARAM_FIR)

        if (get_bits1(gbp))

            if (read_filter_params(m, gbp, ch, FIR) < 0)

                return -1;



    if (s->param_presence_flags & PARAM_IIR)

        if (get_bits1(gbp))

            if (read_filter_params(m, gbp, ch, IIR) < 0)

                return -1;



    if (fir->order && iir->order &&

        fir->shift != iir->shift) {

        av_log(m->avctx, AV_LOG_ERROR,

                "FIR and IIR filters must use the same precision.\n");

        return -1;

    }

    /* The FIR and IIR filters must have the same precision.

        * To simplify the filtering code, only the precision of the

        * FIR filter is considered. If only the IIR filter is employed,

        * the FIR filter precision is set to that of the IIR filter, so

        * that the filtering code can use it. */

    if (!fir->order && iir->order)

        fir->shift = iir->shift;



    if (s->param_presence_flags & PARAM_HUFFOFFSET)

        if (get_bits1(gbp))

            cp->huff_offset = get_sbits(gbp, 15);



    cp->codebook  = get_bits(gbp, 2);

    cp->huff_lsbs = get_bits(gbp, 5);



    cp->sign_huff_offset = calculate_sign_huff(m, substr, ch);



    /* TODO: validate */



    return 0;

}
