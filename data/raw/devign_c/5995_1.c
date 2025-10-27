static void gen_bitops (DisasContext *ctx, uint32_t opc, int rt,

                        int rs, int lsb, int msb)

{

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();



    gen_load_gpr(t1, rs);

    switch (opc) {

    case OPC_EXT:

        if (lsb + msb > 31)

            goto fail;

        tcg_gen_shri_tl(t0, t1, lsb);

        if (msb != 31) {

            tcg_gen_andi_tl(t0, t0, (1 << (msb + 1)) - 1);

        } else {

            tcg_gen_ext32s_tl(t0, t0);

        }

        break;

#if defined(TARGET_MIPS64)

    case OPC_DEXTM:

        tcg_gen_shri_tl(t0, t1, lsb);

        if (msb != 31) {

            tcg_gen_andi_tl(t0, t0, (1ULL << (msb + 1 + 32)) - 1);

        }

        break;

    case OPC_DEXTU:

        tcg_gen_shri_tl(t0, t1, lsb + 32);

        tcg_gen_andi_tl(t0, t0, (1ULL << (msb + 1)) - 1);

        break;

    case OPC_DEXT:

        tcg_gen_shri_tl(t0, t1, lsb);

        tcg_gen_andi_tl(t0, t0, (1ULL << (msb + 1)) - 1);

        break;

#endif

    case OPC_INS:

        if (lsb > msb)

            goto fail;

        gen_load_gpr(t0, rt);

        tcg_gen_deposit_tl(t0, t0, t1, lsb, msb - lsb + 1);

        tcg_gen_ext32s_tl(t0, t0);

        break;

#if defined(TARGET_MIPS64)

    case OPC_DINSM:

        gen_load_gpr(t0, rt);

        tcg_gen_deposit_tl(t0, t0, t1, lsb, msb + 32 - lsb + 1);

        break;

    case OPC_DINSU:

        gen_load_gpr(t0, rt);

        tcg_gen_deposit_tl(t0, t0, t1, lsb + 32, msb - lsb + 1);

        break;

    case OPC_DINS:

        gen_load_gpr(t0, rt);

        tcg_gen_deposit_tl(t0, t0, t1, lsb, msb - lsb + 1);

        break;

#endif

    default:

fail:

        MIPS_INVAL("bitops");

        generate_exception(ctx, EXCP_RI);

        tcg_temp_free(t0);

        tcg_temp_free(t1);

        return;

    }

    gen_store_gpr(t0, rt);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
