static inline void gen_efdnabs(DisasContext *ctx)

{

    if (unlikely(!ctx->spe_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_APU);

        return;

    }

#if defined(TARGET_PPC64)

    tcg_gen_ori_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)], 0x8000000000000000LL);

#else

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_ori_tl(cpu_gprh[rD(ctx->opcode)], cpu_gprh[rA(ctx->opcode)], 0x80000000);

#endif

}
