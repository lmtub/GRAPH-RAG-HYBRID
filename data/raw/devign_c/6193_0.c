static void gen_loongson_integer(DisasContext *ctx, uint32_t opc,

                                 int rd, int rs, int rt)

{

    const char *opn = "loongson";

    TCGv t0, t1;



    if (rd == 0) {

        /* Treat as NOP. */

        MIPS_DEBUG("NOP");

        return;

    }



    switch (opc) {

    case OPC_MULT_G_2E:

    case OPC_MULT_G_2F:

    case OPC_MULTU_G_2E:

    case OPC_MULTU_G_2F:

#if defined(TARGET_MIPS64)

    case OPC_DMULT_G_2E:

    case OPC_DMULT_G_2F:

    case OPC_DMULTU_G_2E:

    case OPC_DMULTU_G_2F:

#endif

        t0 = tcg_temp_new();

        t1 = tcg_temp_new();

        break;

    default:

        t0 = tcg_temp_local_new();

        t1 = tcg_temp_local_new();

        break;

    }



    gen_load_gpr(t0, rs);

    gen_load_gpr(t1, rt);



    switch (opc) {

    case OPC_MULT_G_2E:

    case OPC_MULT_G_2F:

        tcg_gen_mul_tl(cpu_gpr[rd], t0, t1);

        tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        opn = "mult.g";

        break;

    case OPC_MULTU_G_2E:

    case OPC_MULTU_G_2F:

        tcg_gen_ext32u_tl(t0, t0);

        tcg_gen_ext32u_tl(t1, t1);

        tcg_gen_mul_tl(cpu_gpr[rd], t0, t1);

        tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        opn = "multu.g";

        break;

    case OPC_DIV_G_2E:

    case OPC_DIV_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            int l3 = gen_new_label();

            tcg_gen_ext32s_tl(t0, t0);

            tcg_gen_ext32s_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l3);

            gen_set_label(l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, INT_MIN, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1, l2);

            tcg_gen_mov_tl(cpu_gpr[rd], t0);

            tcg_gen_br(l3);

            gen_set_label(l2);

            tcg_gen_div_tl(cpu_gpr[rd], t0, t1);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

            gen_set_label(l3);

        }

        opn = "div.g";

        break;

    case OPC_DIVU_G_2E:

    case OPC_DIVU_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_divu_tl(cpu_gpr[rd], t0, t1);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

            gen_set_label(l2);

        }

        opn = "divu.g";

        break;

    case OPC_MOD_G_2E:

    case OPC_MOD_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            int l3 = gen_new_label();

            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, INT_MIN, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1, l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l3);

            gen_set_label(l2);

            tcg_gen_rem_tl(cpu_gpr[rd], t0, t1);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

            gen_set_label(l3);

        }

        opn = "mod.g";

        break;

    case OPC_MODU_G_2E:

    case OPC_MODU_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_remu_tl(cpu_gpr[rd], t0, t1);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

            gen_set_label(l2);

        }

        opn = "modu.g";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DMULT_G_2E:

    case OPC_DMULT_G_2F:

        tcg_gen_mul_tl(cpu_gpr[rd], t0, t1);

        opn = "dmult.g";

        break;

    case OPC_DMULTU_G_2E:

    case OPC_DMULTU_G_2F:

        tcg_gen_mul_tl(cpu_gpr[rd], t0, t1);

        opn = "dmultu.g";

        break;

    case OPC_DDIV_G_2E:

    case OPC_DDIV_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            int l3 = gen_new_label();

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l3);

            gen_set_label(l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, -1LL << 63, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1LL, l2);

            tcg_gen_mov_tl(cpu_gpr[rd], t0);

            tcg_gen_br(l3);

            gen_set_label(l2);

            tcg_gen_div_tl(cpu_gpr[rd], t0, t1);

            gen_set_label(l3);

        }

        opn = "ddiv.g";

        break;

    case OPC_DDIVU_G_2E:

    case OPC_DDIVU_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_divu_tl(cpu_gpr[rd], t0, t1);

            gen_set_label(l2);

        }

        opn = "ddivu.g";

        break;

    case OPC_DMOD_G_2E:

    case OPC_DMOD_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            int l3 = gen_new_label();

            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            tcg_gen_brcondi_tl(TCG_COND_NE, t0, -1LL << 63, l2);

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, -1LL, l2);

            gen_set_label(l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l3);

            gen_set_label(l2);

            tcg_gen_rem_tl(cpu_gpr[rd], t0, t1);

            gen_set_label(l3);

        }

        opn = "dmod.g";

        break;

    case OPC_DMODU_G_2E:

    case OPC_DMODU_G_2F:

        {

            int l1 = gen_new_label();

            int l2 = gen_new_label();

            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

            tcg_gen_br(l2);

            gen_set_label(l1);

            tcg_gen_remu_tl(cpu_gpr[rd], t0, t1);

            gen_set_label(l2);

        }

        opn = "dmodu.g";

        break;

#endif

    }



    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %s", opn, regnames[rd], regnames[rs]);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
