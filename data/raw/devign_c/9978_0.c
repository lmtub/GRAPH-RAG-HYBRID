static inline void gen_evsel (DisasContext *ctx)

{

    if (unlikely(!ctx->spe_enabled)) {

        RET_EXCP(ctx, EXCP_NO_SPE, 0);

        return;

    }

    gen_op_load_crf_T0(ctx->opcode & 0x7);

    gen_op_load_gpr64_T0(rA(ctx->opcode));

    gen_op_load_gpr64_T1(rB(ctx->opcode));

    gen_op_evsel();

    gen_op_store_T0_gpr64(rD(ctx->opcode));

}
