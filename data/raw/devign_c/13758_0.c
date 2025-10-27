static void gen_conditional_store(DisasContext *ctx, TCGv EA,

                                  int reg, int size)

{

    int l1;



    tcg_gen_trunc_tl_i32(cpu_crf[0], cpu_so);

    l1 = gen_new_label();

    tcg_gen_brcond_tl(TCG_COND_NE, EA, cpu_reserve, l1);

    tcg_gen_ori_i32(cpu_crf[0], cpu_crf[0], 1 << CRF_EQ);

#if defined(TARGET_PPC64)

    if (size == 8) {

        gen_qemu_st64(ctx, cpu_gpr[reg], EA);

    } else

#endif

    if (size == 4) {

        gen_qemu_st32(ctx, cpu_gpr[reg], EA);

    } else if (size == 2) {

        gen_qemu_st16(ctx, cpu_gpr[reg], EA);

#if defined(TARGET_PPC64)

    } else if (size == 16) {

        TCGv gpr1, gpr2 , EA8;

        if (unlikely(ctx->le_mode)) {

            gpr1 = cpu_gpr[reg+1];

            gpr2 = cpu_gpr[reg];

        } else {

            gpr1 = cpu_gpr[reg];

            gpr2 = cpu_gpr[reg+1];

        }

        gen_qemu_st64(ctx, gpr1, EA);

        EA8 = tcg_temp_local_new();

        gen_addr_add(ctx, EA8, EA, 8);

        gen_qemu_st64(ctx, gpr2, EA8);

        tcg_temp_free(EA8);

#endif

    } else {

        gen_qemu_st8(ctx, cpu_gpr[reg], EA);

    }

    gen_set_label(l1);

    tcg_gen_movi_tl(cpu_reserve, -1);

}
