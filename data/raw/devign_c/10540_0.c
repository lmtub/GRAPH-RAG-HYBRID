static void gen_arith (CPUState *env, DisasContext *ctx, uint32_t opc,

                       int rd, int rs, int rt)

{

    const char *opn = "arith";

    TCGv t0 = tcg_temp_local_new(TCG_TYPE_TL);

    TCGv t1 = tcg_temp_local_new(TCG_TYPE_TL);



    if (rd == 0 && opc != OPC_ADD && opc != OPC_SUB

       && opc != OPC_DADD && opc != OPC_DSUB) {

        /* If no destination, treat it as a NOP.

           For add & sub, we must generate the overflow exception when needed. */

        MIPS_DEBUG("NOP");

        goto out;

    }

    gen_load_gpr(t0, rs);

    /* Specialcase the conventional move operation. */

    if (rt == 0 && (opc == OPC_ADDU || opc == OPC_DADDU

                    || opc == OPC_SUBU || opc == OPC_DSUBU)) {

        gen_store_gpr(t0, rd);

        goto out;

    }

    gen_load_gpr(t1, rt);

    switch (opc) {

    case OPC_ADD:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_ext32s_tl(r_tmp1, t0);

            tcg_gen_ext32s_tl(r_tmp2, t1);

            tcg_gen_add_tl(t0, r_tmp1, r_tmp2);



            tcg_gen_xor_tl(r_tmp1, r_tmp1, t1);

            tcg_gen_xori_tl(r_tmp1, r_tmp1, -1);

            tcg_gen_xor_tl(r_tmp2, t0, t1);

            tcg_gen_and_tl(r_tmp1, r_tmp1, r_tmp2);

            tcg_temp_free(r_tmp2);

            tcg_gen_shri_tl(r_tmp1, r_tmp1, 31);

            tcg_gen_brcondi_tl(TCG_COND_EQ, r_tmp1, 0, l1);

            tcg_temp_free(r_tmp1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);



            tcg_gen_ext32s_tl(t0, t0);

        }

        opn = "add";

        break;

    case OPC_ADDU:

        tcg_gen_ext32s_tl(t0, t0);

        tcg_gen_ext32s_tl(t1, t1);

        tcg_gen_add_tl(t0, t0, t1);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "addu";

        break;

    case OPC_SUB:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_ext32s_tl(r_tmp1, t0);

            tcg_gen_ext32s_tl(r_tmp2, t1);

            tcg_gen_sub_tl(t0, r_tmp1, r_tmp2);



            tcg_gen_xor_tl(r_tmp2, r_tmp1, t1);

            tcg_gen_xor_tl(r_tmp1, r_tmp1, t0);

            tcg_gen_and_tl(r_tmp1, r_tmp1, r_tmp2);

            tcg_temp_free(r_tmp2);

            tcg_gen_shri_tl(r_tmp1, r_tmp1, 31);

            tcg_gen_brcondi_tl(TCG_COND_EQ, r_tmp1, 0, l1);

            tcg_temp_free(r_tmp1);

            /* operands of different sign, first operand and result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);



            tcg_gen_ext32s_tl(t0, t0);

        }

        opn = "sub";

        break;

    case OPC_SUBU:

        tcg_gen_ext32s_tl(t0, t0);

        tcg_gen_ext32s_tl(t1, t1);

        tcg_gen_sub_tl(t0, t0, t1);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "subu";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DADD:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_mov_tl(r_tmp1, t0);

            tcg_gen_add_tl(t0, t0, t1);



            tcg_gen_xor_tl(r_tmp1, r_tmp1, t1);

            tcg_gen_xori_tl(r_tmp1, r_tmp1, -1);

            tcg_gen_xor_tl(r_tmp2, t0, t1);

            tcg_gen_and_tl(r_tmp1, r_tmp1, r_tmp2);

            tcg_temp_free(r_tmp2);

            tcg_gen_shri_tl(r_tmp1, r_tmp1, 63);

            tcg_gen_brcondi_tl(TCG_COND_EQ, r_tmp1, 0, l1);

            tcg_temp_free(r_tmp1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

        }

        opn = "dadd";

        break;

    case OPC_DADDU:

        tcg_gen_add_tl(t0, t0, t1);

        opn = "daddu";

        break;

    case OPC_DSUB:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_mov_tl(r_tmp1, t0);

            tcg_gen_sub_tl(t0, t0, t1);



            tcg_gen_xor_tl(r_tmp2, r_tmp1, t1);

            tcg_gen_xor_tl(r_tmp1, r_tmp1, t0);

            tcg_gen_and_tl(r_tmp1, r_tmp1, r_tmp2);

            tcg_temp_free(r_tmp2);

            tcg_gen_shri_tl(r_tmp1, r_tmp1, 63);

            tcg_gen_brcondi_tl(TCG_COND_EQ, r_tmp1, 0, l1);

            tcg_temp_free(r_tmp1);

            /* operands of different sign, first operand and result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

        }

        opn = "dsub";

        break;

    case OPC_DSUBU:

        tcg_gen_sub_tl(t0, t0, t1);

        opn = "dsubu";

        break;

#endif

    case OPC_SLT:

        gen_op_lt(t0, t1);

        opn = "slt";

        break;

    case OPC_SLTU:

        gen_op_ltu(t0, t1);

        opn = "sltu";

        break;

    case OPC_AND:

        tcg_gen_and_tl(t0, t0, t1);

        opn = "and";

        break;

    case OPC_NOR:

        tcg_gen_or_tl(t0, t0, t1);

        tcg_gen_not_tl(t0, t0);

        opn = "nor";

        break;

    case OPC_OR:

        tcg_gen_or_tl(t0, t0, t1);

        opn = "or";

        break;

    case OPC_XOR:

        tcg_gen_xor_tl(t0, t0, t1);

        opn = "xor";

        break;

    case OPC_MUL:

        tcg_gen_ext32s_tl(t0, t0);

        tcg_gen_ext32s_tl(t1, t1);

        tcg_gen_mul_tl(t0, t0, t1);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "mul";

        break;

    case OPC_MOVN:

        {

            int l1 = gen_new_label();



            tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

            gen_store_gpr(t0, rd);

            gen_set_label(l1);

        }

        opn = "movn";

        goto print;

    case OPC_MOVZ:

        {

            int l1 = gen_new_label();



            tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

            gen_store_gpr(t0, rd);

            gen_set_label(l1);

        }

        opn = "movz";

        goto print;

    case OPC_SLLV:

        tcg_gen_ext32u_tl(t0, t0);

        tcg_gen_ext32u_tl(t1, t1);

        tcg_gen_andi_tl(t0, t0, 0x1f);

        tcg_gen_shl_tl(t0, t1, t0);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "sllv";

        break;

    case OPC_SRAV:

        tcg_gen_ext32s_tl(t1, t1);

        tcg_gen_andi_tl(t0, t0, 0x1f);

        tcg_gen_sar_tl(t0, t1, t0);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "srav";

        break;

    case OPC_SRLV:

        switch ((ctx->opcode >> 6) & 0x1f) {

        case 0:

            tcg_gen_ext32u_tl(t1, t1);

            tcg_gen_andi_tl(t0, t0, 0x1f);

            tcg_gen_shr_tl(t0, t1, t0);

            tcg_gen_ext32s_tl(t0, t0);

            opn = "srlv";

            break;

        case 1:

            /* rotrv is decoded as srlv on non-R2 CPUs */

            if (env->insn_flags & ISA_MIPS32R2) {

                int l1 = gen_new_label();

                int l2 = gen_new_label();



                tcg_gen_andi_tl(t0, t0, 0x1f);

                tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 0, l1);

                {

                    TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);

                    TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_I32);



                    tcg_gen_trunc_tl_i32(r_tmp1, t0);

                    tcg_gen_trunc_tl_i32(r_tmp2, t1);

                    tcg_gen_rotr_i32(r_tmp1, r_tmp1, r_tmp2);

                    tcg_temp_free(r_tmp1);

                    tcg_temp_free(r_tmp2);

                    tcg_gen_br(l2);

                }

                gen_set_label(l1);

                tcg_gen_mov_tl(t0, t1);

                gen_set_label(l2);

                opn = "rotrv";

            } else {

                tcg_gen_ext32u_tl(t1, t1);

                tcg_gen_andi_tl(t0, t0, 0x1f);

                tcg_gen_shr_tl(t0, t1, t0);

                tcg_gen_ext32s_tl(t0, t0);

                opn = "srlv";

            }

            break;

        default:

            MIPS_INVAL("invalid srlv flag");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

#if defined(TARGET_MIPS64)

    case OPC_DSLLV:

        tcg_gen_andi_tl(t0, t0, 0x3f);

        tcg_gen_shl_tl(t0, t1, t0);

        opn = "dsllv";

        break;

    case OPC_DSRAV:

        tcg_gen_andi_tl(t0, t0, 0x3f);

        tcg_gen_sar_tl(t0, t1, t0);

        opn = "dsrav";

        break;

    case OPC_DSRLV:

        switch ((ctx->opcode >> 6) & 0x1f) {

        case 0:

            tcg_gen_andi_tl(t0, t0, 0x3f);

            tcg_gen_shr_tl(t0, t1, t0);

            opn = "dsrlv";

            break;

        case 1:

            /* drotrv is decoded as dsrlv on non-R2 CPUs */

            if (env->insn_flags & ISA_MIPS32R2) {

                int l1 = gen_new_label();

                int l2 = gen_new_label();



                tcg_gen_andi_tl(t0, t0, 0x3f);

                tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 0, l1);

                {

                    tcg_gen_rotr_tl(t0, t1, t0);

                    tcg_gen_br(l2);

                }

                gen_set_label(l1);

                tcg_gen_mov_tl(t0, t1);

                gen_set_label(l2);

                opn = "drotrv";

            } else {

                tcg_gen_andi_tl(t0, t0, 0x3f);

                tcg_gen_shr_tl(t0, t1, t0);

                opn = "dsrlv";

            }

            break;

        default:

            MIPS_INVAL("invalid dsrlv flag");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

#endif

    default:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        goto out;

    }

    gen_store_gpr(t0, rd);

 print:

    MIPS_DEBUG("%s %s, %s, %s", opn, regnames[rd], regnames[rs], regnames[rt]);

 out:

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
