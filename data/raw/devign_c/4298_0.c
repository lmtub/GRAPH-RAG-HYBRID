static void gen_arith_imm (CPUState *env, DisasContext *ctx, uint32_t opc,

                           int rt, int rs, int16_t imm)

{

    target_ulong uimm;

    const char *opn = "imm arith";

    TCGv t0 = tcg_temp_local_new(TCG_TYPE_TL);



    if (rt == 0 && opc != OPC_ADDI && opc != OPC_DADDI) {

        /* If no destination, treat it as a NOP.

           For addi, we must generate the overflow exception when needed. */

        MIPS_DEBUG("NOP");

        goto out;

    }

    uimm = (uint16_t)imm;

    switch (opc) {

    case OPC_ADDI:

    case OPC_ADDIU:

#if defined(TARGET_MIPS64)

    case OPC_DADDI:

    case OPC_DADDIU:

#endif

    case OPC_SLTI:

    case OPC_SLTIU:

        uimm = (target_long)imm; /* Sign extend to 32/64 bits */

        /* Fall through. */

    case OPC_ANDI:

    case OPC_ORI:

    case OPC_XORI:

        gen_load_gpr(t0, rs);

        break;

    case OPC_LUI:

        tcg_gen_movi_tl(t0, imm << 16);

        break;

    case OPC_SLL:

    case OPC_SRA:

    case OPC_SRL:

#if defined(TARGET_MIPS64)

    case OPC_DSLL:

    case OPC_DSRA:

    case OPC_DSRL:

    case OPC_DSLL32:

    case OPC_DSRA32:

    case OPC_DSRL32:

#endif

        uimm &= 0x1f;

        gen_load_gpr(t0, rs);

        break;

    }

    switch (opc) {

    case OPC_ADDI:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_ext32s_tl(r_tmp1, t0);

            tcg_gen_addi_tl(t0, r_tmp1, uimm);



            tcg_gen_xori_tl(r_tmp1, r_tmp1, uimm);

            tcg_gen_xori_tl(r_tmp1, r_tmp1, -1);

            tcg_gen_xori_tl(r_tmp2, t0, uimm);

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

        opn = "addi";

        break;

    case OPC_ADDIU:

        tcg_gen_ext32s_tl(t0, t0);

        tcg_gen_addi_tl(t0, t0, uimm);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "addiu";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DADDI:

        {

            TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_TL);

            TCGv r_tmp2 = tcg_temp_new(TCG_TYPE_TL);

            int l1 = gen_new_label();



            save_cpu_state(ctx, 1);

            tcg_gen_mov_tl(r_tmp1, t0);

            tcg_gen_addi_tl(t0, t0, uimm);



            tcg_gen_xori_tl(r_tmp1, r_tmp1, uimm);

            tcg_gen_xori_tl(r_tmp1, r_tmp1, -1);

            tcg_gen_xori_tl(r_tmp2, t0, uimm);

            tcg_gen_and_tl(r_tmp1, r_tmp1, r_tmp2);

            tcg_temp_free(r_tmp2);

            tcg_gen_shri_tl(r_tmp1, r_tmp1, 63);

            tcg_gen_brcondi_tl(TCG_COND_EQ, r_tmp1, 0, l1);

            tcg_temp_free(r_tmp1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

        }

        opn = "daddi";

        break;

    case OPC_DADDIU:

        tcg_gen_addi_tl(t0, t0, uimm);

        opn = "daddiu";

        break;

#endif

    case OPC_SLTI:

        gen_op_lti(t0, uimm);

        opn = "slti";

        break;

    case OPC_SLTIU:

        gen_op_ltiu(t0, uimm);

        opn = "sltiu";

        break;

    case OPC_ANDI:

        tcg_gen_andi_tl(t0, t0, uimm);

        opn = "andi";

        break;

    case OPC_ORI:

        tcg_gen_ori_tl(t0, t0, uimm);

        opn = "ori";

        break;

    case OPC_XORI:

        tcg_gen_xori_tl(t0, t0, uimm);

        opn = "xori";

        break;

    case OPC_LUI:

        opn = "lui";

        break;

    case OPC_SLL:

        tcg_gen_ext32u_tl(t0, t0);

        tcg_gen_shli_tl(t0, t0, uimm);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "sll";

        break;

    case OPC_SRA:

        tcg_gen_ext32s_tl(t0, t0);

        tcg_gen_sari_tl(t0, t0, uimm);

        tcg_gen_ext32s_tl(t0, t0);

        opn = "sra";

        break;

    case OPC_SRL:

        switch ((ctx->opcode >> 21) & 0x1f) {

        case 0:

            tcg_gen_ext32u_tl(t0, t0);

            tcg_gen_shri_tl(t0, t0, uimm);

            tcg_gen_ext32s_tl(t0, t0);

            opn = "srl";

            break;

        case 1:

            /* rotr is decoded as srl on non-R2 CPUs */

            if (env->insn_flags & ISA_MIPS32R2) {

                if (uimm != 0) {

                    TCGv r_tmp1 = tcg_temp_new(TCG_TYPE_I32);



                    tcg_gen_trunc_tl_i32(r_tmp1, t0);

                    tcg_gen_rotri_i32(r_tmp1, r_tmp1, uimm);

                    tcg_gen_ext_i32_tl(t0, r_tmp1);

                    tcg_temp_free(r_tmp1);

                }

                opn = "rotr";

            } else {

                tcg_gen_ext32u_tl(t0, t0);

                tcg_gen_shri_tl(t0, t0, uimm);

                tcg_gen_ext32s_tl(t0, t0);

                opn = "srl";

            }

            break;

        default:

            MIPS_INVAL("invalid srl flag");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

#if defined(TARGET_MIPS64)

    case OPC_DSLL:

        tcg_gen_shli_tl(t0, t0, uimm);

        opn = "dsll";

        break;

    case OPC_DSRA:

        tcg_gen_sari_tl(t0, t0, uimm);

        opn = "dsra";

        break;

    case OPC_DSRL:

        switch ((ctx->opcode >> 21) & 0x1f) {

        case 0:

            tcg_gen_shri_tl(t0, t0, uimm);

            opn = "dsrl";

            break;

        case 1:

            /* drotr is decoded as dsrl on non-R2 CPUs */

            if (env->insn_flags & ISA_MIPS32R2) {

                if (uimm != 0) {

                    tcg_gen_rotri_tl(t0, t0, uimm);

                }

                opn = "drotr";

            } else {

                tcg_gen_shri_tl(t0, t0, uimm);

                opn = "dsrl";

            }

            break;

        default:

            MIPS_INVAL("invalid dsrl flag");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_DSLL32:

        tcg_gen_shli_tl(t0, t0, uimm + 32);

        opn = "dsll32";

        break;

    case OPC_DSRA32:

        tcg_gen_sari_tl(t0, t0, uimm + 32);

        opn = "dsra32";

        break;

    case OPC_DSRL32:

        switch ((ctx->opcode >> 21) & 0x1f) {

        case 0:

            tcg_gen_shri_tl(t0, t0, uimm + 32);

            opn = "dsrl32";

            break;

        case 1:

            /* drotr32 is decoded as dsrl32 on non-R2 CPUs */

            if (env->insn_flags & ISA_MIPS32R2) {

                tcg_gen_rotri_tl(t0, t0, uimm + 32);

                opn = "drotr32";

            } else {

                tcg_gen_shri_tl(t0, t0, uimm + 32);

                opn = "dsrl32";

            }

            break;

        default:

            MIPS_INVAL("invalid dsrl32 flag");

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

    gen_store_gpr(t0, rt);

    MIPS_DEBUG("%s %s, %s, " TARGET_FMT_lx, opn, regnames[rt], regnames[rs], uimm);

 out:

    tcg_temp_free(t0);

}
