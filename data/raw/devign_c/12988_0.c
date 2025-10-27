static void compute_exp_strategy(AC3EncodeContext *s)

{

    int ch, blk, blk1;



    for (ch = !s->cpl_on; ch <= s->fbw_channels; ch++) {

        uint8_t *exp_strategy = s->exp_strategy[ch];

        uint8_t *exp          = s->blocks[0].exp[ch];

        int exp_diff;



        /* estimate if the exponent variation & decide if they should be

           reused in the next frame */

        exp_strategy[0] = EXP_NEW;

        exp += AC3_MAX_COEFS;

        for (blk = 1; blk < AC3_MAX_BLOCKS; blk++, exp += AC3_MAX_COEFS) {

            if ((ch == CPL_CH && (!s->blocks[blk].cpl_in_use || !s->blocks[blk-1].cpl_in_use)) ||

                (ch  > CPL_CH && (s->blocks[blk].channel_in_cpl[ch] != s->blocks[blk-1].channel_in_cpl[ch]))) {

                exp_strategy[blk] = EXP_NEW;

                continue;

            }

            exp_diff = s->dsp.sad[0](NULL, exp, exp - AC3_MAX_COEFS, 16, 16);

            exp_strategy[blk] = EXP_REUSE;

            if (ch == CPL_CH && exp_diff > (EXP_DIFF_THRESHOLD * (s->blocks[blk].end_freq[ch] - s->start_freq[ch]) / AC3_MAX_COEFS))

                exp_strategy[blk] = EXP_NEW;

            else if (ch > CPL_CH && exp_diff > EXP_DIFF_THRESHOLD)

                exp_strategy[blk] = EXP_NEW;

        }



        /* now select the encoding strategy type : if exponents are often

           recoded, we use a coarse encoding */

        blk = 0;

        while (blk < AC3_MAX_BLOCKS) {

            blk1 = blk + 1;

            while (blk1 < AC3_MAX_BLOCKS && exp_strategy[blk1] == EXP_REUSE)

                blk1++;

            switch (blk1 - blk) {

            case 1:  exp_strategy[blk] = EXP_D45; break;

            case 2:

            case 3:  exp_strategy[blk] = EXP_D25; break;

            default: exp_strategy[blk] = EXP_D15; break;

            }

            blk = blk1;

        }

    }

    if (s->lfe_on) {

        ch = s->lfe_channel;

        s->exp_strategy[ch][0] = EXP_D15;

        for (blk = 1; blk < AC3_MAX_BLOCKS; blk++)

            s->exp_strategy[ch][blk] = EXP_REUSE;

    }

}
