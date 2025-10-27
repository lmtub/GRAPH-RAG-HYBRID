static void gen_ld (CPUState *env, DisasContext *ctx, uint32_t opc,

                    int rt, int base, int16_t offset)

{

    const char *opn = "ld";

    TCGv t0, t1;



    if (rt == 0 && env->insn_flags & (INSN_LOONGSON2E | INSN_LOONGSON2F)) {

        /* Loongson CPU uses a load to zero register for prefetch.

           We emulate it as a NOP. On other CPU we must perform the

           actual memory access. */

        MIPS_DEBUG("NOP");

        return;

    }



    t0 = tcg_temp_new();

    t1 = tcg_temp_new();

    gen_base_offset_addr(ctx, t0, base, offset);



    switch (opc) {

#if defined(TARGET_MIPS64)

    case OPC_LWU:

        save_cpu_state(ctx, 0);

        op_ld_lwu(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lwu";

        break;

    case OPC_LD:

        save_cpu_state(ctx, 0);

        op_ld_ld(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "ld";

        break;

    case OPC_LLD:

        save_cpu_state(ctx, 0);

        op_ld_lld(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lld";

        break;

    case OPC_LDL:

        save_cpu_state(ctx, 1);

        gen_load_gpr(t1, rt);

        gen_helper_3i(ldl, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        opn = "ldl";

        break;

    case OPC_LDR:

        save_cpu_state(ctx, 1);

        gen_load_gpr(t1, rt);

        gen_helper_3i(ldr, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        opn = "ldr";

        break;

    case OPC_LDPC:

        save_cpu_state(ctx, 1);

        tcg_gen_movi_tl(t1, pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        op_ld_ld(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "ldpc";

        break;

#endif

    case OPC_LWPC:

        save_cpu_state(ctx, 1);

        tcg_gen_movi_tl(t1, pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        op_ld_lw(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lwpc";

        break;

    case OPC_LW:

        save_cpu_state(ctx, 0);

        op_ld_lw(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lw";

        break;

    case OPC_LH:

        save_cpu_state(ctx, 0);

        op_ld_lh(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lh";

        break;

    case OPC_LHU:

        save_cpu_state(ctx, 0);

        op_ld_lhu(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lhu";

        break;

    case OPC_LB:

        save_cpu_state(ctx, 0);

        op_ld_lb(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lb";

        break;

    case OPC_LBU:

        save_cpu_state(ctx, 0);

        op_ld_lbu(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lbu";

        break;

    case OPC_LWL:

        save_cpu_state(ctx, 1);

        gen_load_gpr(t1, rt);

        gen_helper_3i(lwl, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        opn = "lwl";

        break;

    case OPC_LWR:

        save_cpu_state(ctx, 1);

        gen_load_gpr(t1, rt);

        gen_helper_3i(lwr, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        opn = "lwr";

        break;

    case OPC_LL:

        save_cpu_state(ctx, 1);

        op_ld_ll(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "ll";

        break;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %d(%s)", opn, regnames[rt], offset, regnames[base]);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
