static int dnxhd_init_rc(DNXHDEncContext *ctx)

{

    CHECKED_ALLOCZ(ctx->mb_rc, 8160*ctx->m.avctx->qmax*sizeof(RCEntry));

    if (ctx->m.avctx->mb_decision != FF_MB_DECISION_RD)

        CHECKED_ALLOCZ(ctx->mb_cmp, ctx->m.mb_num*sizeof(RCCMPEntry));



    ctx->frame_bits = (ctx->cid_table->coding_unit_size - 640 - 4) * 8;

    ctx->qscale = 1;

    ctx->lambda = 2<<LAMBDA_FRAC_BITS; // qscale 2

    return 0;

 fail:

    return -1;

}
