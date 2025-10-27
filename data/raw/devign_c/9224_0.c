static int lag_read_prob_header(lag_rac *rac, GetBitContext *gb)

{

    int i, j, scale_factor;

    unsigned prob, cumulative_target;

    unsigned cumul_prob = 0;

    unsigned scaled_cumul_prob = 0;



    rac->prob[0] = 0;

    rac->prob[257] = UINT_MAX;

    /* Read probabilities from bitstream */

    for (i = 1; i < 257; i++) {

        if (lag_decode_prob(gb, &rac->prob[i]) < 0) {

            av_log(rac->avctx, AV_LOG_ERROR, "Invalid probability encountered.\n");

            return -1;

        }

        if ((uint64_t)cumul_prob + rac->prob[i] > UINT_MAX) {

            av_log(rac->avctx, AV_LOG_ERROR, "Integer overflow encountered in cumulative probability calculation.\n");

            return -1;

        }

        cumul_prob += rac->prob[i];

        if (!rac->prob[i]) {

            if (lag_decode_prob(gb, &prob)) {

                av_log(rac->avctx, AV_LOG_ERROR, "Invalid probability run encountered.\n");

                return -1;

            }

            if (prob > 257 - i)

                prob = 257 - i;

            for (j = 0; j < prob; j++)

                rac->prob[++i] = 0;

        }

    }



    if (!cumul_prob) {

        av_log(rac->avctx, AV_LOG_ERROR, "All probabilities are 0!\n");

        return -1;

    }



    /* Scale probabilities so cumulative probability is an even power of 2. */

    scale_factor = av_log2(cumul_prob);



    if (cumul_prob & (cumul_prob - 1)) {

        uint64_t mul = softfloat_reciprocal(cumul_prob);

        for (i = 1; i < 257; i++) {

            rac->prob[i] = softfloat_mul(rac->prob[i], mul);

            scaled_cumul_prob += rac->prob[i];

        }



        scale_factor++;

        cumulative_target = 1 << scale_factor;



        if (scaled_cumul_prob > cumulative_target) {

            av_log(rac->avctx, AV_LOG_ERROR,

                   "Scaled probabilities are larger than target!\n");

            return -1;

        }



        scaled_cumul_prob = cumulative_target - scaled_cumul_prob;



        for (i = 1; scaled_cumul_prob; i = (i & 0x7f) + 1) {

            if (rac->prob[i]) {

                rac->prob[i]++;

                scaled_cumul_prob--;

            }

            /* Comment from reference source:

             * if (b & 0x80 == 0) {     // order of operations is 'wrong'; it has been left this way

             *                          // since the compression change is negligable and fixing it

             *                          // breaks backwards compatibility

             *      b =- (signed int)b;

             *      b &= 0xFF;

             * } else {

             *      b++;

             *      b &= 0x7f;

             * }

             */

        }

    }



    rac->scale = scale_factor;



    /* Fill probability array with cumulative probability for each symbol. */

    for (i = 1; i < 257; i++)

        rac->prob[i] += rac->prob[i - 1];



    return 0;

}
