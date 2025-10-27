static int decode_subframe_lpc(ShortenContext *s, int command, int channel,

                               int residual_size, int32_t coffset)

{

    int pred_order, sum, qshift, init_sum, i, j;

    const int *coeffs;



    if (command == FN_QLPC) {

        /* read/validate prediction order */

        pred_order = get_ur_golomb_shorten(&s->gb, LPCQSIZE);

        if (pred_order > s->nwrap) {

            av_log(s->avctx, AV_LOG_ERROR, "invalid pred_order %d\n",

                   pred_order);

            return AVERROR(EINVAL);

        }

        /* read LPC coefficients */

        for (i = 0; i < pred_order; i++)

            s->coeffs[i] = get_sr_golomb_shorten(&s->gb, LPCQUANT);

        coeffs = s->coeffs;



        qshift = LPCQUANT;

    } else {

        /* fixed LPC coeffs */

        pred_order = command;

        if (pred_order >= FF_ARRAY_ELEMS(fixed_coeffs)) {

            av_log(s->avctx, AV_LOG_ERROR, "invalid pred_order %d\n",

                   pred_order);

            return AVERROR_INVALIDDATA;

        }

        coeffs     = fixed_coeffs[pred_order];

        qshift     = 0;

    }



    /* subtract offset from previous samples to use in prediction */

    if (command == FN_QLPC && coffset)

        for (i = -pred_order; i < 0; i++)

            s->decoded[channel][i] -= coffset;



    /* decode residual and do LPC prediction */

    init_sum = pred_order ? (command == FN_QLPC ? s->lpcqoffset : 0) : coffset;

    for (i = 0; i < s->blocksize; i++) {

        sum = init_sum;

        for (j = 0; j < pred_order; j++)

            sum += coeffs[j] * s->decoded[channel][i - j - 1];

        s->decoded[channel][i] = get_sr_golomb_shorten(&s->gb, residual_size) +

                                 (sum >> qshift);

    }



    /* add offset to current samples */

    if (command == FN_QLPC && coffset)

        for (i = 0; i < s->blocksize; i++)

            s->decoded[channel][i] += coffset;



    return 0;

}
