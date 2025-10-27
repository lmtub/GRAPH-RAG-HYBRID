static inline int decode_subframe(FLACContext *s, int channel)

{

    int type, wasted = 0;

    int i, tmp;



    s->curr_bps = s->bps;

    if (channel == 0) {

        if (s->decorrelation == RIGHT_SIDE)

            s->curr_bps++;

    } else {

        if (s->decorrelation == LEFT_SIDE || s->decorrelation == MID_SIDE)

            s->curr_bps++;

    }

    if (s->curr_bps > 32) {

        ff_log_missing_feature(s->avctx, "decorrelated bit depth > 32", 0);

        return -1;

    }



    if (get_bits1(&s->gb)) {

        av_log(s->avctx, AV_LOG_ERROR, "invalid subframe padding\n");

        return -1;

    }

    type = get_bits(&s->gb, 6);



    if (get_bits1(&s->gb)) {

        wasted = 1;

        while (!get_bits1(&s->gb))

            wasted++;

        s->curr_bps -= wasted;

    }



//FIXME use av_log2 for types

    if (type == 0) {

        tmp = get_sbits_long(&s->gb, s->curr_bps);

        for (i = 0; i < s->blocksize; i++)

            s->decoded[channel][i] = tmp;

    } else if (type == 1) {

        for (i = 0; i < s->blocksize; i++)

            s->decoded[channel][i] = get_sbits_long(&s->gb, s->curr_bps);

    } else if ((type >= 8) && (type <= 12)) {

        if (decode_subframe_fixed(s, channel, type & ~0x8) < 0)

            return -1;

    } else if (type >= 32) {

        if (decode_subframe_lpc(s, channel, (type & ~0x20)+1) < 0)

            return -1;

    } else {

        av_log(s->avctx, AV_LOG_ERROR, "invalid coding type\n");

        return -1;

    }



    if (wasted) {

        int i;

        for (i = 0; i < s->blocksize; i++)

            s->decoded[channel][i] <<= wasted;

    }



    return 0;

}
