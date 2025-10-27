static int dnxhd_init_qmat(DNXHDEncContext *ctx, int lbias, int cbias)

{

    // init first elem to 1 to avoid div by 0 in convert_matrix

    uint16_t weight_matrix[64] = {1,}; // convert_matrix needs uint16_t*

    int qscale, i;



    CHECKED_ALLOCZ(ctx->qmatrix_l,   (ctx->m.avctx->qmax+1) * 64 * sizeof(int));

    CHECKED_ALLOCZ(ctx->qmatrix_c,   (ctx->m.avctx->qmax+1) * 64 * sizeof(int));

    CHECKED_ALLOCZ(ctx->qmatrix_l16, (ctx->m.avctx->qmax+1) * 64 * 2 * sizeof(uint16_t));

    CHECKED_ALLOCZ(ctx->qmatrix_c16, (ctx->m.avctx->qmax+1) * 64 * 2 * sizeof(uint16_t));



    for (i = 1; i < 64; i++) {

        int j = ctx->m.dsp.idct_permutation[ff_zigzag_direct[i]];

        weight_matrix[j] = ctx->cid_table->luma_weight[i];

    }

    ff_convert_matrix(&ctx->m.dsp, ctx->qmatrix_l, ctx->qmatrix_l16, weight_matrix,

                      ctx->m.intra_quant_bias, 1, ctx->m.avctx->qmax, 1);

    for (i = 1; i < 64; i++) {

        int j = ctx->m.dsp.idct_permutation[ff_zigzag_direct[i]];

        weight_matrix[j] = ctx->cid_table->chroma_weight[i];

    }

    ff_convert_matrix(&ctx->m.dsp, ctx->qmatrix_c, ctx->qmatrix_c16, weight_matrix,

                      ctx->m.intra_quant_bias, 1, ctx->m.avctx->qmax, 1);

    for (qscale = 1; qscale <= ctx->m.avctx->qmax; qscale++) {

        for (i = 0; i < 64; i++) {

            ctx->qmatrix_l  [qscale]   [i] <<= 2; ctx->qmatrix_c  [qscale]   [i] <<= 2;

            ctx->qmatrix_l16[qscale][0][i] <<= 2; ctx->qmatrix_l16[qscale][1][i] <<= 2;

            ctx->qmatrix_c16[qscale][0][i] <<= 2; ctx->qmatrix_c16[qscale][1][i] <<= 2;

        }

    }

    return 0;

 fail:

    return -1;

}
