static int ra144_decode_frame(AVCodecContext * avctx, void *vdata,

                              int *data_size, const uint8_t *buf, int buf_size)

{

    static const uint8_t sizes[10] = {6, 5, 5, 4, 4, 3, 3, 3, 3, 2};

    unsigned int refl_rms[4];    // RMS of the reflection coefficients

    uint16_t block_coefs[4][30]; // LPC coefficients of each sub-block

    unsigned int lpc_refl[10];   // LPC reflection coefficients of the frame

    int i, c;

    int16_t *data = vdata;

    unsigned int energy;



    RA144Context *ractx = avctx->priv_data;

    GetBitContext gb;



    if(buf_size < 20) {

        av_log(avctx, AV_LOG_ERROR,

               "Frame too small (%d bytes). Truncated file?\n", buf_size);

        *data_size = 0;

        return buf_size;

    }

    init_get_bits(&gb, buf, 20 * 8);



    for (i=0; i<10; i++)

        lpc_refl[i] = lpc_refl_cb[i][get_bits(&gb, sizes[i])];



    eval_coefs(ractx->lpc_coef[0], lpc_refl);

    ractx->lpc_refl_rms[0] = rms(lpc_refl);



    energy = energy_tab[get_bits(&gb, 5)];



    refl_rms[0] = interp(ractx, block_coefs[0], 0, 1, ractx->old_energy);

    refl_rms[1] = interp(ractx, block_coefs[1], 1, energy <= ractx->old_energy,

                    t_sqrt(energy*ractx->old_energy) >> 12);

    refl_rms[2] = interp(ractx, block_coefs[2], 2, 0, energy);

    refl_rms[3] = rescale_rms(ractx->lpc_refl_rms[0], energy);



    int_to_int16(block_coefs[3], ractx->lpc_coef[0]);



    for (c=0; c<4; c++) {

        do_output_subblock(ractx, block_coefs[c], refl_rms[c], &gb);



        for (i=0; i<BLOCKSIZE; i++)

            *data++ = av_clip_int16(ractx->curr_sblock[i + 10] << 2);

    }



    ractx->old_energy = energy;

    ractx->lpc_refl_rms[1] = ractx->lpc_refl_rms[0];



    FFSWAP(unsigned int *, ractx->lpc_coef[0], ractx->lpc_coef[1]);



    *data_size = 2*160;

    return 20;

}
