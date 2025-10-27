static void gen_wait(DisasContext *ctx)

{

    TCGv_i32 t0 = tcg_temp_new_i32();

    tcg_gen_st_i32(t0, cpu_env,

                   -offsetof(PowerPCCPU, env) + offsetof(CPUState, halted));

    tcg_temp_free_i32(t0);

    /* Stop translation, as the CPU is supposed to sleep from now */

    gen_exception_err(ctx, EXCP_HLT, 1);

}
