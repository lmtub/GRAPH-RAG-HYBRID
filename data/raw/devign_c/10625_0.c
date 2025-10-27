static int read_filter_params(MLPDecodeContext *m, GetBitContext *gbp,

                              unsigned int channel, unsigned int filter)

{

    FilterParams *fp = &m->channel_params[channel].filter_params[filter];

    const int max_order = filter ? MAX_IIR_ORDER : MAX_FIR_ORDER;

    const char fchar = filter ? 'I' : 'F';

    int i, order;



    // Filter is 0 for FIR, 1 for IIR.

    assert(filter < 2);



    m->filter_changed[channel][filter]++;



    order = get_bits(gbp, 4);

    if (order > max_order) {

        av_log(m->avctx, AV_LOG_ERROR,

               "%cIR filter order %d is greater than maximum %d.\n",

               fchar, order, max_order);

        return -1;

    }

    fp->order = order;



    if (order > 0) {

        int coeff_bits, coeff_shift;



        fp->shift = get_bits(gbp, 4);



        coeff_bits  = get_bits(gbp, 5);

        coeff_shift = get_bits(gbp, 3);

        if (coeff_bits < 1 || coeff_bits > 16) {

            av_log(m->avctx, AV_LOG_ERROR,

                   "%cIR filter coeff_bits must be between 1 and 16.\n",

                   fchar);

            return -1;

        }

        if (coeff_bits + coeff_shift > 16) {

            av_log(m->avctx, AV_LOG_ERROR,

                   "Sum of coeff_bits and coeff_shift for %cIR filter must be 16 or less.\n",

                   fchar);

            return -1;

        }



        for (i = 0; i < order; i++)

            fp->coeff[i] = get_sbits(gbp, coeff_bits) << coeff_shift;



        if (get_bits1(gbp)) {

            int state_bits, state_shift;



            if (filter == FIR) {

                av_log(m->avctx, AV_LOG_ERROR,

                       "FIR filter has state data specified.\n");

                return -1;

            }



            state_bits  = get_bits(gbp, 4);

            state_shift = get_bits(gbp, 4);



            /* TODO: Check validity of state data. */



            for (i = 0; i < order; i++)

                fp->state[i] = get_sbits(gbp, state_bits) << state_shift;

        }

    }



    return 0;

}
