static void filter_channel(MLPDecodeContext *m, unsigned int substr,

                           unsigned int channel)

{

    SubStream *s = &m->substream[substr];

    int32_t firbuf[MAX_BLOCKSIZE + MAX_FIR_ORDER];

    int32_t iirbuf[MAX_BLOCKSIZE + MAX_IIR_ORDER];

    FilterParams *fir = &m->channel_params[channel].filter_params[FIR];

    FilterParams *iir = &m->channel_params[channel].filter_params[IIR];

    unsigned int filter_shift = fir->shift;

    int32_t mask = MSB_MASK(s->quant_step_size[channel]);

    int index = MAX_BLOCKSIZE;

    int i;



    memcpy(&firbuf[MAX_BLOCKSIZE], &fir->state[0],

            MAX_FIR_ORDER * sizeof(int32_t));

    memcpy(&iirbuf[MAX_BLOCKSIZE], &iir->state[0],

            MAX_IIR_ORDER * sizeof(int32_t));



    for (i = 0; i < s->blocksize; i++) {

        int32_t residual = m->sample_buffer[i + s->blockpos][channel];

        unsigned int order;

        int64_t accum = 0;

        int32_t result;



        /* TODO: Move this code to DSPContext? */



        for (order = 0; order < fir->order; order++)

            accum += (int64_t)firbuf[index + order] *

                                fir->coeff[order];

        for (order = 0; order < iir->order; order++)

            accum += (int64_t)iirbuf[index + order] *

                                iir->coeff[order];



        accum  = accum >> filter_shift;

        result = (accum + residual) & mask;



        --index;



        firbuf[index] = result;

        iirbuf[index] = result - accum;



        m->sample_buffer[i + s->blockpos][channel] = result;

    }



    memcpy(&fir->state[0], &firbuf[index],

            MAX_FIR_ORDER * sizeof(int32_t));

    memcpy(&iir->state[0], &iirbuf[index],

            MAX_IIR_ORDER * sizeof(int32_t));

}
