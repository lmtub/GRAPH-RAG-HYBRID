static int decode_subframe_fixed(FLACContext *s, int channel, int pred_order)

{

    const int blocksize = s->blocksize;

    int32_t *decoded = s->decoded[channel];

    int a, b, c, d, i;



    /* warm up samples */

    for (i = 0; i < pred_order; i++) {

        decoded[i] = get_sbits(&s->gb, s->curr_bps);

    }



    if (decode_residuals(s, channel, pred_order) < 0)

        return -1;



    if (pred_order > 0)

        a = decoded[pred_order-1];

    if (pred_order > 1)

        b = a - decoded[pred_order-2];

    if (pred_order > 2)

        c = b - decoded[pred_order-2] + decoded[pred_order-3];

    if (pred_order > 3)

        d = c - decoded[pred_order-2] + 2*decoded[pred_order-3] - decoded[pred_order-4];



    switch (pred_order) {

    case 0:

        break;

    case 1:

        for (i = pred_order; i < blocksize; i++)

            decoded[i] = a += decoded[i];

        break;

    case 2:

        for (i = pred_order; i < blocksize; i++)

            decoded[i] = a += b += decoded[i];

        break;

    case 3:

        for (i = pred_order; i < blocksize; i++)

            decoded[i] = a += b += c += decoded[i];

        break;

    case 4:

        for (i = pred_order; i < blocksize; i++)

            decoded[i] = a += b += c += d += decoded[i];

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "illegal pred order %d\n", pred_order);

        return -1;

    }



    return 0;

}
