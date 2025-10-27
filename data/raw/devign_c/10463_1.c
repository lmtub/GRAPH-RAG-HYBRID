static FFPsyWindowInfo psy_lame_window(FFPsyContext *ctx, const float *audio,

                                       const float *la, int channel, int prev_type)

{

    AacPsyContext *pctx = (AacPsyContext*) ctx->model_priv_data;

    AacPsyChannel *pch  = &pctx->ch[channel];

    int grouping     = 0;

    int uselongblock = 1;

    int attacks[AAC_NUM_BLOCKS_SHORT + 1] = { 0 };

    float clippings[AAC_NUM_BLOCKS_SHORT];

    int i;

    FFPsyWindowInfo wi = { { 0 } };



    if (la) {

        float hpfsmpl[AAC_BLOCK_SIZE_LONG];

        float const *pf = hpfsmpl;

        float attack_intensity[(AAC_NUM_BLOCKS_SHORT + 1) * PSY_LAME_NUM_SUBBLOCKS];

        float energy_subshort[(AAC_NUM_BLOCKS_SHORT + 1) * PSY_LAME_NUM_SUBBLOCKS];

        float energy_short[AAC_NUM_BLOCKS_SHORT + 1] = { 0 };

        const float *firbuf = la + (AAC_BLOCK_SIZE_SHORT/4 - PSY_LAME_FIR_LEN);

        int att_sum = 0;



        /* LAME comment: apply high pass filter of fs/4 */

        psy_hp_filter(firbuf, hpfsmpl, psy_fir_coeffs);



        /* Calculate the energies of each sub-shortblock */

        for (i = 0; i < PSY_LAME_NUM_SUBBLOCKS; i++) {

            energy_subshort[i] = pch->prev_energy_subshort[i + ((AAC_NUM_BLOCKS_SHORT - 1) * PSY_LAME_NUM_SUBBLOCKS)];

            assert(pch->prev_energy_subshort[i + ((AAC_NUM_BLOCKS_SHORT - 2) * PSY_LAME_NUM_SUBBLOCKS + 1)] > 0);

            attack_intensity[i] = energy_subshort[i] / pch->prev_energy_subshort[i + ((AAC_NUM_BLOCKS_SHORT - 2) * PSY_LAME_NUM_SUBBLOCKS + 1)];

            energy_short[0] += energy_subshort[i];

        }



        for (i = 0; i < AAC_NUM_BLOCKS_SHORT * PSY_LAME_NUM_SUBBLOCKS; i++) {

            float const *const pfe = pf + AAC_BLOCK_SIZE_LONG / (AAC_NUM_BLOCKS_SHORT * PSY_LAME_NUM_SUBBLOCKS);

            float p = 1.0f;

            for (; pf < pfe; pf++)

                p = FFMAX(p, fabsf(*pf));

            pch->prev_energy_subshort[i] = energy_subshort[i + PSY_LAME_NUM_SUBBLOCKS] = p;

            energy_short[1 + i / PSY_LAME_NUM_SUBBLOCKS] += p;

            /* NOTE: The indexes below are [i + 3 - 2] in the LAME source.

             *       Obviously the 3 and 2 have some significance, or this would be just [i + 1]

             *       (which is what we use here). What the 3 stands for is ambiguous, as it is both

             *       number of short blocks, and the number of sub-short blocks.

             *       It seems that LAME is comparing each sub-block to sub-block + 1 in the

             *       previous block.

             */

            if (p > energy_subshort[i + 1])

                p = p / energy_subshort[i + 1];

            else if (energy_subshort[i + 1] > p * 10.0f)

                p = energy_subshort[i + 1] / (p * 10.0f);

            else

                p = 0.0;

            attack_intensity[i + PSY_LAME_NUM_SUBBLOCKS] = p;

        }



        /* compare energy between sub-short blocks */

        for (i = 0; i < (AAC_NUM_BLOCKS_SHORT + 1) * PSY_LAME_NUM_SUBBLOCKS; i++)

            if (!attacks[i / PSY_LAME_NUM_SUBBLOCKS])

                if (attack_intensity[i] > pch->attack_threshold)

                    attacks[i / PSY_LAME_NUM_SUBBLOCKS] = (i % PSY_LAME_NUM_SUBBLOCKS) + 1;



        /* should have energy change between short blocks, in order to avoid periodic signals */

        /* Good samples to show the effect are Trumpet test songs */

        /* GB: tuned (1) to avoid too many short blocks for test sample TRUMPET */

        /* RH: tuned (2) to let enough short blocks through for test sample FSOL and SNAPS */

        for (i = 1; i < AAC_NUM_BLOCKS_SHORT + 1; i++) {

            float const u = energy_short[i - 1];

            float const v = energy_short[i];

            float const m = FFMAX(u, v);

            if (m < 40000) {                          /* (2) */

                if (u < 1.7f * v && v < 1.7f * u) {   /* (1) */

                    if (i == 1 && attacks[0] < attacks[i])

                        attacks[0] = 0;

                    attacks[i] = 0;

                }

            }

            att_sum += attacks[i];

        }



        if (attacks[0] <= pch->prev_attack)

            attacks[0] = 0;



        att_sum += attacks[0];

        /* 3 below indicates the previous attack happened in the last sub-block of the previous sequence */

        if (pch->prev_attack == 3 || att_sum) {

            uselongblock = 0;



            for (i = 1; i < AAC_NUM_BLOCKS_SHORT + 1; i++)

                if (attacks[i] && attacks[i-1])

                    attacks[i] = 0;

        }

    } else {

        /* We have no lookahead info, so just use same type as the previous sequence. */

        uselongblock = !(prev_type == EIGHT_SHORT_SEQUENCE);

    }



    lame_apply_block_type(pch, &wi, uselongblock);



    /* Calculate input sample maximums and evaluate clipping risk */

    if (audio) {

        for (i = 0; i < AAC_NUM_BLOCKS_SHORT; i++) {

            const float *wbuf = audio + i * AAC_BLOCK_SIZE_SHORT;

            float max = 0;

            int j;

            for (j = 0; j < AAC_BLOCK_SIZE_SHORT; j++)

                max = FFMAX(max, fabsf(wbuf[j]));

            clippings[i] = max;

        }

    } else {

        for (i = 0; i < 8; i++)

            clippings[i] = 0;

    }



    wi.window_type[1] = prev_type;

    if (wi.window_type[0] != EIGHT_SHORT_SEQUENCE) {

        float clipping = 0.0f;



        wi.num_windows  = 1;

        wi.grouping[0]  = 1;

        if (wi.window_type[0] == LONG_START_SEQUENCE)

            wi.window_shape = 0;

        else

            wi.window_shape = 1;



        for (i = 0; i < 8; i++)

            clipping = FFMAX(clipping, clippings[i]);

        wi.clipping[0] = clipping;

    } else {

        int lastgrp = 0;



        wi.num_windows = 8;

        wi.window_shape = 0;

        for (i = 0; i < 8; i++) {

            if (!((pch->next_grouping >> i) & 1))

                lastgrp = i;

            wi.grouping[lastgrp]++;

        }



        for (i = 0; i < 8; i += wi.grouping[i]) {

            int w;

            float clipping = 0.0f;

            for (w = 0; w < wi.grouping[i] && !clipping; w++)

                clipping = FFMAX(clipping, clippings[i+w]);

            wi.clipping[i] = clipping;

        }

    }



    /* Determine grouping, based on the location of the first attack, and save for

     * the next frame.

     * FIXME: Move this to analysis.

     * TODO: Tune groupings depending on attack location

     * TODO: Handle more than one attack in a group

     */

    for (i = 0; i < 9; i++) {

        if (attacks[i]) {

            grouping = i;

            break;

        }

    }

    pch->next_grouping = window_grouping[grouping];



    pch->prev_attack = attacks[8];



    return wi;

}
