static int twinvq_read_bitstream(AVCodecContext *avctx, TwinVQContext *tctx,

                                 const uint8_t *buf, int buf_size)

{

    TwinVQFrameData     *bits = &tctx->bits;

    const TwinVQModeTab *mtab = tctx->mtab;

    int channels              = tctx->avctx->channels;

    int sub;

    GetBitContext gb;

    int i, j, k;



    if (buf_size * 8 < avctx->bit_rate * mtab->size / avctx->sample_rate + 8) {

        av_log(avctx, AV_LOG_ERROR,

               "Frame too small (%d bytes). Truncated file?\n", buf_size);

        return AVERROR(EINVAL);

    }



    init_get_bits(&gb, buf, buf_size * 8);

    skip_bits(&gb, get_bits(&gb, 8));



    bits->window_type = get_bits(&gb, TWINVQ_WINDOW_TYPE_BITS);



    if (bits->window_type > 8) {

        av_log(avctx, AV_LOG_ERROR, "Invalid window type, broken sample?\n");

        return AVERROR_INVALIDDATA;

    }



    bits->ftype = ff_twinvq_wtype_to_ftype_table[tctx->bits.window_type];



    sub = mtab->fmode[bits->ftype].sub;



    read_cb_data(tctx, &gb, bits->main_coeffs, bits->ftype);



    for (i = 0; i < channels; i++)

        for (j = 0; j < sub; j++)

            for (k = 0; k < mtab->fmode[bits->ftype].bark_n_coef; k++)

                bits->bark1[i][j][k] =

                    get_bits(&gb, mtab->fmode[bits->ftype].bark_n_bit);



    for (i = 0; i < channels; i++)

        for (j = 0; j < sub; j++)

            bits->bark_use_hist[i][j] = get_bits1(&gb);



    if (bits->ftype == TWINVQ_FT_LONG) {

        for (i = 0; i < channels; i++)

            bits->gain_bits[i] = get_bits(&gb, TWINVQ_GAIN_BITS);

    } else {

        for (i = 0; i < channels; i++) {

            bits->gain_bits[i] = get_bits(&gb, TWINVQ_GAIN_BITS);

            for (j = 0; j < sub; j++)

                bits->sub_gain_bits[i * sub + j] = get_bits(&gb,

                                                       TWINVQ_SUB_GAIN_BITS);

        }

    }



    for (i = 0; i < channels; i++) {

        bits->lpc_hist_idx[i] = get_bits(&gb, mtab->lsp_bit0);

        bits->lpc_idx1[i]     = get_bits(&gb, mtab->lsp_bit1);



        for (j = 0; j < mtab->lsp_split; j++)

            bits->lpc_idx2[i][j] = get_bits(&gb, mtab->lsp_bit2);

    }



    if (bits->ftype == TWINVQ_FT_LONG) {

        read_cb_data(tctx, &gb, bits->ppc_coeffs, 3);

        for (i = 0; i < channels; i++) {

            bits->p_coef[i] = get_bits(&gb, mtab->ppc_period_bit);

            bits->g_coef[i] = get_bits(&gb, mtab->pgain_bit);

        }

    }



    return 0;

}
