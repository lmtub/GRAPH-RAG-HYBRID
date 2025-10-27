static int decode_residuals(FLACContext *s, int channel, int pred_order)

{

    int i, tmp, partition, method_type, rice_order;

    int sample = 0, samples;



    method_type = get_bits(&s->gb, 2);

    if (method_type > 1) {

        av_log(s->avctx, AV_LOG_ERROR, "illegal residual coding method %d\n",

               method_type);

        return -1;

    }



    rice_order = get_bits(&s->gb, 4);



    samples= s->blocksize >> rice_order;

    if (pred_order > samples) {

        av_log(s->avctx, AV_LOG_ERROR, "invalid predictor order: %i > %i\n",

               pred_order, samples);

        return -1;

    }



    sample=

    i= pred_order;

    for (partition = 0; partition < (1 << rice_order); partition++) {

        tmp = get_bits(&s->gb, method_type == 0 ? 4 : 5);

        if (tmp == (method_type == 0 ? 15 : 31)) {

            tmp = get_bits(&s->gb, 5);

            for (; i < samples; i++, sample++)

                s->decoded[channel][sample] = get_sbits_long(&s->gb, tmp);

        } else {

            for (; i < samples; i++, sample++) {

                s->decoded[channel][sample] = get_sr_golomb_flac(&s->gb, tmp, INT_MAX, 0);

            }

        }

        i= 0;

    }



    return 0;

}
