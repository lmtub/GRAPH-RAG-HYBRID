static void gen_ld (CPUMIPSState *env, DisasContext *ctx, uint32_t opc,

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

    gen_base_offset_addr(ctx, t0, base, offset);



    switch (opc) {

#if defined(TARGET_MIPS64)

    case OPC_LWU:

        tcg_gen_qemu_ld32u(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lwu";

        break;

    case OPC_LD:

        tcg_gen_qemu_ld64(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "ld";

        break;

    case OPC_LLD:

        save_cpu_state(ctx, 1);

        op_ld_lld(t0, t0, ctx);

        gen_store_gpr(t0, rt);

        opn = "lld";

        break;

    case OPC_LDL:

        save_cpu_state(ctx, 1);

        t1 = tcg_temp_new();

        gen_load_gpr(t1, rt);

        gen_helper_1e2i(ldl, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        tcg_temp_free(t1);

        opn = "ldl";

        break;

    case OPC_LDR:

        save_cpu_state(ctx, 1);

        t1 = tcg_temp_new();

        gen_load_gpr(t1, rt);

        gen_helper_1e2i(ldr, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        tcg_temp_free(t1);

        opn = "ldr";

        break;

    case OPC_LDPC:

        t1 = tcg_const_tl(pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_qemu_ld64(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "ldpc";

        break;

#endif

    case OPC_LWPC:

        t1 = tcg_const_tl(pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_qemu_ld32s(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lwpc";

        break;

    case OPC_LW:

        tcg_gen_qemu_ld32s(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lw";

        break;

    case OPC_LH:

        tcg_gen_qemu_ld16s(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lh";

        break;

    case OPC_LHU:

        tcg_gen_qemu_ld16u(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lhu";

        break;

    case OPC_LB:

        tcg_gen_qemu_ld8s(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lb";

        break;

    case OPC_LBU:

        tcg_gen_qemu_ld8u(t0, t0, ctx->mem_idx);

        gen_store_gpr(t0, rt);

        opn = "lbu";

        break;

    case OPC_LWL:

        save_cpu_state(ctx, 1);

        t1 = tcg_temp_new();

        gen_load_gpr(t1, rt);

        gen_helper_1e2i(lwl, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        tcg_temp_free(t1);

        opn = "lwl";

        break;

    case OPC_LWR:

        save_cpu_state(ctx, 1);

        t1 = tcg_temp_new();

        gen_load_gpr(t1, rt);

        gen_helper_1e2i(lwr, t1, t1, t0, ctx->mem_idx);

        gen_store_gpr(t1, rt);

        tcg_temp_free(t1);

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

}
