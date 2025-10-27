static void gen_dstst(DisasContext *ctx)

{

    if (rA(ctx->opcode) == 0) {

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_LSWX);

    } else {

        /* interpreted as no-op */

    }



}
