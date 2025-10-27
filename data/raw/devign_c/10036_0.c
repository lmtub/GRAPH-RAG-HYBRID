static inline void gen_evmergehilo(DisasContext *ctx)

{

    if (unlikely(!ctx->spe_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_APU);

        return;

    }

#if defined(TARGET_PPC64)

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    tcg_gen_ext32u_tl(t0, cpu_gpr[rB(ctx->opcode)]);

    tcg_gen_andi_tl(t1, cpu_gpr[rA(ctx->opcode)], 0xFFFFFFFF0000000ULL);

    tcg_gen_or_tl(cpu_gpr[rD(ctx->opcode)], t0, t1);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

#else

    tcg_gen_mov_i32(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rB(ctx->opcode)]);

    tcg_gen_mov_i32(cpu_gprh[rD(ctx->opcode)], cpu_gprh[rA(ctx->opcode)]);

#endif

}
