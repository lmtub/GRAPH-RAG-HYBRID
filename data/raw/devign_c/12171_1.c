static av_unused FFPsyWindowInfo psy_3gpp_window(FFPsyContext *ctx,

                                                 const int16_t *audio,

                                                 const int16_t *la,

                                                 int channel, int prev_type)

{

    int i, j;

    int br               = ctx->avctx->bit_rate / ctx->avctx->channels;

    int attack_ratio     = br <= 16000 ? 18 : 10;

    AacPsyContext *pctx = (AacPsyContext*) ctx->model_priv_data;

    AacPsyChannel *pch  = &pctx->ch[channel];

    uint8_t grouping     = 0;

    int next_type        = pch->next_window_seq;

    FFPsyWindowInfo wi  = { { 0 } };



    if (la) {

        float s[8], v;

        int switch_to_eight = 0;

        float sum = 0.0, sum2 = 0.0;

        int attack_n = 0;

        int stay_short = 0;

        for (i = 0; i < 8; i++) {

            for (j = 0; j < 128; j++) {

                v = iir_filter(la[i*128+j], pch->iir_state);

                sum += v*v;

            }

            s[i]  = sum;

            sum2 += sum;

        }

        for (i = 0; i < 8; i++) {

            if (s[i] > pch->win_energy * attack_ratio) {

                attack_n        = i + 1;

                switch_to_eight = 1;

                break;

            }

        }

        pch->win_energy = pch->win_energy*7/8 + sum2/64;



        wi.window_type[1] = prev_type;

        switch (prev_type) {

        case ONLY_LONG_SEQUENCE:

            wi.window_type[0] = switch_to_eight ? LONG_START_SEQUENCE : ONLY_LONG_SEQUENCE;

            next_type = switch_to_eight ? EIGHT_SHORT_SEQUENCE : ONLY_LONG_SEQUENCE;

            break;

        case LONG_START_SEQUENCE:

            wi.window_type[0] = EIGHT_SHORT_SEQUENCE;

            grouping = pch->next_grouping;

            next_type = switch_to_eight ? EIGHT_SHORT_SEQUENCE : LONG_STOP_SEQUENCE;

            break;

        case LONG_STOP_SEQUENCE:

            wi.window_type[0] = switch_to_eight ? LONG_START_SEQUENCE : ONLY_LONG_SEQUENCE;

            next_type = switch_to_eight ? EIGHT_SHORT_SEQUENCE : ONLY_LONG_SEQUENCE;

            break;

        case EIGHT_SHORT_SEQUENCE:

            stay_short = next_type == EIGHT_SHORT_SEQUENCE || switch_to_eight;

            wi.window_type[0] = stay_short ? EIGHT_SHORT_SEQUENCE : LONG_STOP_SEQUENCE;

            grouping = next_type == EIGHT_SHORT_SEQUENCE ? pch->next_grouping : 0;

            next_type = switch_to_eight ? EIGHT_SHORT_SEQUENCE : LONG_STOP_SEQUENCE;

            break;

        }



        pch->next_grouping = window_grouping[attack_n];

        pch->next_window_seq = next_type;

    } else {

        for (i = 0; i < 3; i++)

            wi.window_type[i] = prev_type;

        grouping = (prev_type == EIGHT_SHORT_SEQUENCE) ? window_grouping[0] : 0;

    }



    wi.window_shape   = 1;

    if (wi.window_type[0] != EIGHT_SHORT_SEQUENCE) {

        wi.num_windows = 1;

        wi.grouping[0] = 1;

    } else {

        int lastgrp = 0;

        wi.num_windows = 8;

        for (i = 0; i < 8; i++) {

            if (!((grouping >> i) & 1))

                lastgrp = i;

            wi.grouping[lastgrp]++;

        }

    }



    return wi;

}
