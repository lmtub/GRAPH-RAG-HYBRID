static int ra144_encode_frame(AVCodecContext *avctx, uint8_t *frame,

                              int buf_size, void *data)

{

    static const uint8_t sizes[LPC_ORDER] = {64, 32, 32, 16, 16, 8, 8, 8, 8, 4};

    static const uint8_t bit_sizes[LPC_ORDER] = {6, 5, 5, 4, 4, 3, 3, 3, 3, 2};

    RA144Context *ractx;

    PutBitContext pb;

    int32_t lpc_data[NBLOCKS * BLOCKSIZE];

    int32_t lpc_coefs[LPC_ORDER][MAX_LPC_ORDER];

    int shift[LPC_ORDER];

    int16_t block_coefs[NBLOCKS][LPC_ORDER];

    int lpc_refl[LPC_ORDER];    /**< reflection coefficients of the frame */

    unsigned int refl_rms[NBLOCKS]; /**< RMS of the reflection coefficients */

    int energy = 0;

    int i, idx;



    if (buf_size < FRAMESIZE) {

        av_log(avctx, AV_LOG_ERROR, "output buffer too small\n");

        return 0;

    }

    ractx = avctx->priv_data;



    /**

     * Since the LPC coefficients are calculated on a frame centered over the

     * fourth subframe, to encode a given frame, data from the next frame is

     * needed. In each call to this function, the previous frame (whose data are

     * saved in the encoder context) is encoded, and data from the current frame

     * are saved in the encoder context to be used in the next function call.

     */

    for (i = 0; i < (2 * BLOCKSIZE + BLOCKSIZE / 2); i++) {

        lpc_data[i] = ractx->curr_block[BLOCKSIZE + BLOCKSIZE / 2 + i];

        energy += (lpc_data[i] * lpc_data[i]) >> 4;

    }

    for (i = 2 * BLOCKSIZE + BLOCKSIZE / 2; i < NBLOCKS * BLOCKSIZE; i++) {

        lpc_data[i] = *((int16_t *)data + i - 2 * BLOCKSIZE - BLOCKSIZE / 2) >>

                      2;

        energy += (lpc_data[i] * lpc_data[i]) >> 4;

    }

    energy = ff_energy_tab[quantize(ff_t_sqrt(energy >> 5) >> 10, ff_energy_tab,

                                    32)];



    ff_lpc_calc_coefs(&ractx->lpc_ctx, lpc_data, NBLOCKS * BLOCKSIZE, LPC_ORDER,

                      LPC_ORDER, 16, lpc_coefs, shift, FF_LPC_TYPE_LEVINSON,

                      0, ORDER_METHOD_EST, 12, 0);

    for (i = 0; i < LPC_ORDER; i++)

        block_coefs[NBLOCKS - 1][i] = -(lpc_coefs[LPC_ORDER - 1][i] <<

                                        (12 - shift[LPC_ORDER - 1]));



    /**

     * TODO: apply perceptual weighting of the input speech through bandwidth

     * expansion of the LPC filter.

     */



    if (ff_eval_refl(lpc_refl, block_coefs[NBLOCKS - 1], avctx)) {

        /**

         * The filter is unstable: use the coefficients of the previous frame.

         */

        ff_int_to_int16(block_coefs[NBLOCKS - 1], ractx->lpc_coef[1]);

        ff_eval_refl(lpc_refl, block_coefs[NBLOCKS - 1], avctx);

    }

    init_put_bits(&pb, frame, buf_size);

    for (i = 0; i < LPC_ORDER; i++) {

        idx = quantize(lpc_refl[i], ff_lpc_refl_cb[i], sizes[i]);

        put_bits(&pb, bit_sizes[i], idx);

        lpc_refl[i] = ff_lpc_refl_cb[i][idx];

    }

    ractx->lpc_refl_rms[0] = ff_rms(lpc_refl);

    ff_eval_coefs(ractx->lpc_coef[0], lpc_refl);

    refl_rms[0] = ff_interp(ractx, block_coefs[0], 1, 1, ractx->old_energy);

    refl_rms[1] = ff_interp(ractx, block_coefs[1], 2,

                            energy <= ractx->old_energy,

                            ff_t_sqrt(energy * ractx->old_energy) >> 12);

    refl_rms[2] = ff_interp(ractx, block_coefs[2], 3, 0, energy);

    refl_rms[3] = ff_rescale_rms(ractx->lpc_refl_rms[0], energy);

    ff_int_to_int16(block_coefs[NBLOCKS - 1], ractx->lpc_coef[0]);

    put_bits(&pb, 5, quantize(energy, ff_energy_tab, 32));

    for (i = 0; i < NBLOCKS; i++)

        ra144_encode_subblock(ractx, ractx->curr_block + i * BLOCKSIZE,

                              block_coefs[i], refl_rms[i], &pb);

    flush_put_bits(&pb);

    ractx->old_energy = energy;

    ractx->lpc_refl_rms[1] = ractx->lpc_refl_rms[0];

    FFSWAP(unsigned int *, ractx->lpc_coef[0], ractx->lpc_coef[1]);

    for (i = 0; i < NBLOCKS * BLOCKSIZE; i++)

        ractx->curr_block[i] = *((int16_t *)data + i) >> 2;

    return FRAMESIZE;

}
