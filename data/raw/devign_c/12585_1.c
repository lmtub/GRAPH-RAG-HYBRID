int decode_block_coeffs(VP56RangeCoder *c, int16_t block[16],

                        uint8_t probs[16][3][NUM_DCT_TOKENS - 1],

                        int i, int zero_nhood, int16_t qmul[2])

{

    uint8_t *token_prob = probs[i][zero_nhood];

    if (!vp56_rac_get_prob_branchy(c, token_prob[0]))   // DCT_EOB

        return 0;

    return decode_block_coeffs_internal(c, block, probs, i, token_prob, qmul);

}
