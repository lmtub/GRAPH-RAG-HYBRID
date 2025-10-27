static void gen_trap (DisasContext *ctx, uint32_t opc,

                      int rs, int rt, int16_t imm)

{

    int cond;

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();



    cond = 0;

    /* Load needed operands */

    switch (opc) {

    case OPC_TEQ:

    case OPC_TGE:

    case OPC_TGEU:

    case OPC_TLT:

    case OPC_TLTU:

    case OPC_TNE:

        /* Compare two registers */

        if (rs != rt) {

            gen_load_gpr(t0, rs);

            gen_load_gpr(t1, rt);

            cond = 1;

        }

        break;

    case OPC_TEQI:

    case OPC_TGEI:

    case OPC_TGEIU:

    case OPC_TLTI:

    case OPC_TLTIU:

    case OPC_TNEI:

        /* Compare register to immediate */

        if (rs != 0 || imm != 0) {

            gen_load_gpr(t0, rs);

            tcg_gen_movi_tl(t1, (int32_t)imm);

            cond = 1;

        }

        break;

    }

    if (cond == 0) {

        switch (opc) {

        case OPC_TEQ:   /* rs == rs */

        case OPC_TEQI:  /* r0 == 0  */

        case OPC_TGE:   /* rs >= rs */

        case OPC_TGEI:  /* r0 >= 0  */

        case OPC_TGEU:  /* rs >= rs unsigned */

        case OPC_TGEIU: /* r0 >= 0  unsigned */

            /* Always trap */

            generate_exception(ctx, EXCP_TRAP);

            break;

        case OPC_TLT:   /* rs < rs           */

        case OPC_TLTI:  /* r0 < 0            */

        case OPC_TLTU:  /* rs < rs unsigned  */

        case OPC_TLTIU: /* r0 < 0  unsigned  */

        case OPC_TNE:   /* rs != rs          */

        case OPC_TNEI:  /* r0 != 0           */

            /* Never trap: treat as NOP. */

            break;

        }

    } else {

        int l1 = gen_new_label();



        switch (opc) {

        case OPC_TEQ:

        case OPC_TEQI:

            tcg_gen_brcond_tl(TCG_COND_NE, t0, t1, l1);

            break;

        case OPC_TGE:

        case OPC_TGEI:

            tcg_gen_brcond_tl(TCG_COND_LT, t0, t1, l1);

            break;

        case OPC_TGEU:

        case OPC_TGEIU:

            tcg_gen_brcond_tl(TCG_COND_LTU, t0, t1, l1);

            break;

        case OPC_TLT:

        case OPC_TLTI:

            tcg_gen_brcond_tl(TCG_COND_GE, t0, t1, l1);

            break;

        case OPC_TLTU:

        case OPC_TLTIU:

            tcg_gen_brcond_tl(TCG_COND_GEU, t0, t1, l1);

            break;

        case OPC_TNE:

        case OPC_TNEI:

            tcg_gen_brcond_tl(TCG_COND_EQ, t0, t1, l1);

            break;

        }

        generate_exception(ctx, EXCP_TRAP);

        gen_set_label(l1);

    }

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
