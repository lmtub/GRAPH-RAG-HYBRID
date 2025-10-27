static void decode_opc (CPUMIPSState *env, DisasContext *ctx, int *is_branch)

{

    int32_t offset;

    int rs, rt, rd, sa;

    uint32_t op, op1, op2;

    int16_t imm;



    /* make sure instructions are on a word boundary */

    if (ctx->pc & 0x3) {

        env->CP0_BadVAddr = ctx->pc;

        generate_exception(ctx, EXCP_AdEL);

        return;

    }



    /* Handle blikely not taken case */

    if ((ctx->hflags & MIPS_HFLAG_BMASK_BASE) == MIPS_HFLAG_BL) {

        int l1 = gen_new_label();



        MIPS_DEBUG("blikely condition (" TARGET_FMT_lx ")", ctx->pc + 4);

        tcg_gen_brcondi_tl(TCG_COND_NE, bcond, 0, l1);

        tcg_gen_movi_i32(hflags, ctx->hflags & ~MIPS_HFLAG_BMASK);

        gen_goto_tb(ctx, 1, ctx->pc + 4);

        gen_set_label(l1);

    }



    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

        tcg_gen_debug_insn_start(ctx->pc);

    }



    op = MASK_OP_MAJOR(ctx->opcode);

    rs = (ctx->opcode >> 21) & 0x1f;

    rt = (ctx->opcode >> 16) & 0x1f;

    rd = (ctx->opcode >> 11) & 0x1f;

    sa = (ctx->opcode >> 6) & 0x1f;

    imm = (int16_t)ctx->opcode;

    switch (op) {

    case OPC_SPECIAL:

        op1 = MASK_SPECIAL(ctx->opcode);

        switch (op1) {

        case OPC_SLL:          /* Shift with immediate */

        case OPC_SRA:

            gen_shift_imm(ctx, op1, rd, rt, sa);

            break;

        case OPC_SRL:

            switch ((ctx->opcode >> 21) & 0x1f) {

            case 1:

                /* rotr is decoded as srl on non-R2 CPUs */

                if (ctx->insn_flags & ISA_MIPS32R2) {

                    op1 = OPC_ROTR;

                }

                /* Fallthrough */

            case 0:

                gen_shift_imm(ctx, op1, rd, rt, sa);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_MOVN:         /* Conditional move */

        case OPC_MOVZ:

            check_insn(ctx, ISA_MIPS4 | ISA_MIPS32 |

                                 INSN_LOONGSON2E | INSN_LOONGSON2F);

            gen_cond_move(ctx, op1, rd, rs, rt);

            break;

        case OPC_ADD ... OPC_SUBU:

            gen_arith(ctx, op1, rd, rs, rt);

            break;

        case OPC_SLLV:         /* Shifts */

        case OPC_SRAV:

            gen_shift(ctx, op1, rd, rs, rt);

            break;

        case OPC_SRLV:

            switch ((ctx->opcode >> 6) & 0x1f) {

            case 1:

                /* rotrv is decoded as srlv on non-R2 CPUs */

                if (ctx->insn_flags & ISA_MIPS32R2) {

                    op1 = OPC_ROTRV;

                }

                /* Fallthrough */

            case 0:

                gen_shift(ctx, op1, rd, rs, rt);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_SLT:          /* Set on less than */

        case OPC_SLTU:

            gen_slt(ctx, op1, rd, rs, rt);

            break;

        case OPC_AND:          /* Logic*/

        case OPC_OR:

        case OPC_NOR:

        case OPC_XOR:

            gen_logic(ctx, op1, rd, rs, rt);

            break;

        case OPC_MULT:

        case OPC_MULTU:

            if (sa) {

                check_insn(ctx, INSN_VR54XX);

                op1 = MASK_MUL_VR54XX(ctx->opcode);

                gen_mul_vr54xx(ctx, op1, rd, rs, rt);

            } else {

                gen_muldiv(ctx, op1, rd & 3, rs, rt);

            }

            break;

        case OPC_DIV:

        case OPC_DIVU:

            gen_muldiv(ctx, op1, 0, rs, rt);

            break;

        case OPC_JR ... OPC_JALR:

            gen_compute_branch(ctx, op1, 4, rs, rd, sa);

            *is_branch = 1;

            break;

        case OPC_TGE ... OPC_TEQ: /* Traps */

        case OPC_TNE:

            gen_trap(ctx, op1, rs, rt, -1);

            break;

        case OPC_MFHI:          /* Move from HI/LO */

        case OPC_MFLO:

            gen_HILO(ctx, op1, rs & 3, rd);

            break;

        case OPC_MTHI:

        case OPC_MTLO:          /* Move to HI/LO */

            gen_HILO(ctx, op1, rd & 3, rs);

            break;

        case OPC_PMON:          /* Pmon entry point, also R4010 selsl */

#ifdef MIPS_STRICT_STANDARD

            MIPS_INVAL("PMON / selsl");

            generate_exception(ctx, EXCP_RI);

#else

            gen_helper_0e0i(pmon, sa);

#endif

            break;

        case OPC_SYSCALL:

            generate_exception(ctx, EXCP_SYSCALL);

            ctx->bstate = BS_STOP;

            break;

        case OPC_BREAK:

            generate_exception(ctx, EXCP_BREAK);

            break;

        case OPC_SPIM:

#ifdef MIPS_STRICT_STANDARD

            MIPS_INVAL("SPIM");

            generate_exception(ctx, EXCP_RI);

#else

           /* Implemented as RI exception for now. */

            MIPS_INVAL("spim (unofficial)");

            generate_exception(ctx, EXCP_RI);

#endif

            break;

        case OPC_SYNC:

            /* Treat as NOP. */

            break;



        case OPC_MOVCI:

            check_insn(ctx, ISA_MIPS4 | ISA_MIPS32);

            if (env->CP0_Config1 & (1 << CP0C1_FP)) {

                check_cp1_enabled(ctx);

                gen_movci(ctx, rd, rs, (ctx->opcode >> 18) & 0x7,

                          (ctx->opcode >> 16) & 1);

            } else {

                generate_exception_err(ctx, EXCP_CpU, 1);

            }

            break;



#if defined(TARGET_MIPS64)

       /* MIPS64 specific opcodes */

        case OPC_DSLL:

        case OPC_DSRA:

        case OPC_DSLL32:

        case OPC_DSRA32:

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_shift_imm(ctx, op1, rd, rt, sa);

            break;

        case OPC_DSRL:

            switch ((ctx->opcode >> 21) & 0x1f) {

            case 1:

                /* drotr is decoded as dsrl on non-R2 CPUs */

                if (ctx->insn_flags & ISA_MIPS32R2) {

                    op1 = OPC_DROTR;

                }

                /* Fallthrough */

            case 0:

                check_insn(ctx, ISA_MIPS3);

                check_mips_64(ctx);

                gen_shift_imm(ctx, op1, rd, rt, sa);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DSRL32:

            switch ((ctx->opcode >> 21) & 0x1f) {

            case 1:

                /* drotr32 is decoded as dsrl32 on non-R2 CPUs */

                if (ctx->insn_flags & ISA_MIPS32R2) {

                    op1 = OPC_DROTR32;

                }

                /* Fallthrough */

            case 0:

                check_insn(ctx, ISA_MIPS3);

                check_mips_64(ctx);

                gen_shift_imm(ctx, op1, rd, rt, sa);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DADD ... OPC_DSUBU:

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_arith(ctx, op1, rd, rs, rt);

            break;

        case OPC_DSLLV:

        case OPC_DSRAV:

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_shift(ctx, op1, rd, rs, rt);

            break;

        case OPC_DSRLV:

            switch ((ctx->opcode >> 6) & 0x1f) {

            case 1:

                /* drotrv is decoded as dsrlv on non-R2 CPUs */

                if (ctx->insn_flags & ISA_MIPS32R2) {

                    op1 = OPC_DROTRV;

                }

                /* Fallthrough */

            case 0:

                check_insn(ctx, ISA_MIPS3);

                check_mips_64(ctx);

                gen_shift(ctx, op1, rd, rs, rt);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DMULT ... OPC_DDIVU:

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_muldiv(ctx, op1, 0, rs, rt);

            break;

#endif

        default:            /* Invalid */

            MIPS_INVAL("special");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_SPECIAL2:

        op1 = MASK_SPECIAL2(ctx->opcode);

        switch (op1) {

        case OPC_MADD ... OPC_MADDU: /* Multiply and add/sub */

        case OPC_MSUB ... OPC_MSUBU:

            check_insn(ctx, ISA_MIPS32);

            gen_muldiv(ctx, op1, rd & 3, rs, rt);

            break;

        case OPC_MUL:

            gen_arith(ctx, op1, rd, rs, rt);

            break;

        case OPC_CLO:

        case OPC_CLZ:

            check_insn(ctx, ISA_MIPS32);

            gen_cl(ctx, op1, rd, rs);

            break;

        case OPC_SDBBP:

            /* XXX: not clear which exception should be raised

             *      when in debug mode...

             */

            check_insn(ctx, ISA_MIPS32);

            if (!(ctx->hflags & MIPS_HFLAG_DM)) {

                generate_exception(ctx, EXCP_DBp);

            } else {

                generate_exception(ctx, EXCP_DBp);

            }

            /* Treat as NOP. */

            break;

        case OPC_DIV_G_2F:

        case OPC_DIVU_G_2F:

        case OPC_MULT_G_2F:

        case OPC_MULTU_G_2F:

        case OPC_MOD_G_2F:

        case OPC_MODU_G_2F:

            check_insn(ctx, INSN_LOONGSON2F);

            gen_loongson_integer(ctx, op1, rd, rs, rt);

            break;

#if defined(TARGET_MIPS64)

        case OPC_DCLO:

        case OPC_DCLZ:

            check_insn(ctx, ISA_MIPS64);

            check_mips_64(ctx);

            gen_cl(ctx, op1, rd, rs);

            break;

        case OPC_DMULT_G_2F:

        case OPC_DMULTU_G_2F:

        case OPC_DDIV_G_2F:

        case OPC_DDIVU_G_2F:

        case OPC_DMOD_G_2F:

        case OPC_DMODU_G_2F:

            check_insn(ctx, INSN_LOONGSON2F);

            gen_loongson_integer(ctx, op1, rd, rs, rt);

            break;

#endif

        default:            /* Invalid */

            MIPS_INVAL("special2");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_SPECIAL3:

        op1 = MASK_SPECIAL3(ctx->opcode);

        switch (op1) {

        case OPC_EXT:

        case OPC_INS:

            check_insn(ctx, ISA_MIPS32R2);

            gen_bitops(ctx, op1, rt, rs, sa, rd);

            break;

        case OPC_BSHFL:

            check_insn(ctx, ISA_MIPS32R2);

            op2 = MASK_BSHFL(ctx->opcode);

            gen_bshfl(ctx, op2, rt, rd);

            break;

        case OPC_RDHWR:

            gen_rdhwr(ctx, rt, rd);

            break;

        case OPC_FORK:

            check_insn(ctx, ASE_MT);

            {

                TCGv t0 = tcg_temp_new();

                TCGv t1 = tcg_temp_new();



                gen_load_gpr(t0, rt);

                gen_load_gpr(t1, rs);

                gen_helper_fork(t0, t1);

                tcg_temp_free(t0);

                tcg_temp_free(t1);

            }

            break;

        case OPC_YIELD:

            check_insn(ctx, ASE_MT);

            {

                TCGv t0 = tcg_temp_new();



                save_cpu_state(ctx, 1);

                gen_load_gpr(t0, rs);

                gen_helper_yield(t0, cpu_env, t0);

                gen_store_gpr(t0, rd);

                tcg_temp_free(t0);

            }

            break;

        case OPC_DIV_G_2E ... OPC_DIVU_G_2E:

        case OPC_MOD_G_2E ... OPC_MODU_G_2E:

        case OPC_MULT_G_2E ... OPC_MULTU_G_2E:

        /* OPC_MULT_G_2E, OPC_ADDUH_QB_DSP, OPC_MUL_PH_DSP have

         * the same mask and op1. */

            if ((ctx->insn_flags & ASE_DSPR2) && (op1 == OPC_MULT_G_2E)) {

                op2 = MASK_ADDUH_QB(ctx->opcode);

                switch (op2) {

                case OPC_ADDUH_QB:

                case OPC_ADDUH_R_QB:

                case OPC_ADDQH_PH:

                case OPC_ADDQH_R_PH:

                case OPC_ADDQH_W:

                case OPC_ADDQH_R_W:

                case OPC_SUBUH_QB:

                case OPC_SUBUH_R_QB:

                case OPC_SUBQH_PH:

                case OPC_SUBQH_R_PH:

                case OPC_SUBQH_W:

                case OPC_SUBQH_R_W:

                    gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                    break;

                case OPC_MUL_PH:

                case OPC_MUL_S_PH:

                case OPC_MULQ_S_W:

                case OPC_MULQ_RS_W:

                    gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 1);

                    break;

                default:

                    MIPS_INVAL("MASK ADDUH.QB");

                    generate_exception(ctx, EXCP_RI);

                    break;

                }

            } else if (ctx->insn_flags & INSN_LOONGSON2E) {

                gen_loongson_integer(ctx, op1, rd, rs, rt);

            } else {

                generate_exception(ctx, EXCP_RI);

            }

            break;

        case OPC_LX_DSP:

            op2 = MASK_LX(ctx->opcode);

            switch (op2) {

#if defined(TARGET_MIPS64)

            case OPC_LDX:

#endif

            case OPC_LBUX:

            case OPC_LHX:

            case OPC_LWX:

                gen_mipsdsp_ld(ctx, op2, rd, rs, rt);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK LX");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_ABSQ_S_PH_DSP:

            op2 = MASK_ABSQ_S_PH(ctx->opcode);

            switch (op2) {

            case OPC_ABSQ_S_QB:

            case OPC_ABSQ_S_PH:

            case OPC_ABSQ_S_W:

            case OPC_PRECEQ_W_PHL:

            case OPC_PRECEQ_W_PHR:

            case OPC_PRECEQU_PH_QBL:

            case OPC_PRECEQU_PH_QBR:

            case OPC_PRECEQU_PH_QBLA:

            case OPC_PRECEQU_PH_QBRA:

            case OPC_PRECEU_PH_QBL:

            case OPC_PRECEU_PH_QBR:

            case OPC_PRECEU_PH_QBLA:

            case OPC_PRECEU_PH_QBRA:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_BITREV:

            case OPC_REPL_QB:

            case OPC_REPLV_QB:

            case OPC_REPL_PH:

            case OPC_REPLV_PH:

                gen_mipsdsp_bitinsn(ctx, op1, op2, rd, rt);

                break;

            default:

                MIPS_INVAL("MASK ABSQ_S.PH");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_ADDU_QB_DSP:

            op2 = MASK_ADDU_QB(ctx->opcode);

            switch (op2) {

            case OPC_ADDQ_PH:

            case OPC_ADDQ_S_PH:

            case OPC_ADDQ_S_W:

            case OPC_ADDU_QB:

            case OPC_ADDU_S_QB:

            case OPC_ADDU_PH:

            case OPC_ADDU_S_PH:

            case OPC_SUBQ_PH:

            case OPC_SUBQ_S_PH:

            case OPC_SUBQ_S_W:

            case OPC_SUBU_QB:

            case OPC_SUBU_S_QB:

            case OPC_SUBU_PH:

            case OPC_SUBU_S_PH:

            case OPC_ADDSC:

            case OPC_ADDWC:

            case OPC_MODSUB:

            case OPC_RADDU_W_QB:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_MULEU_S_PH_QBL:

            case OPC_MULEU_S_PH_QBR:

            case OPC_MULQ_RS_PH:

            case OPC_MULEQ_S_W_PHL:

            case OPC_MULEQ_S_W_PHR:

            case OPC_MULQ_S_PH:

                gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 1);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK ADDU.QB");

                generate_exception(ctx, EXCP_RI);

                break;



            }

            break;

        case OPC_CMPU_EQ_QB_DSP:

            op2 = MASK_CMPU_EQ_QB(ctx->opcode);

            switch (op2) {

            case OPC_PRECR_SRA_PH_W:

            case OPC_PRECR_SRA_R_PH_W:

                gen_mipsdsp_arith(ctx, op1, op2, rt, rs, rd);

                break;

            case OPC_PRECR_QB_PH:

            case OPC_PRECRQ_QB_PH:

            case OPC_PRECRQ_PH_W:

            case OPC_PRECRQ_RS_PH_W:

            case OPC_PRECRQU_S_QB_PH:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_CMPU_EQ_QB:

            case OPC_CMPU_LT_QB:

            case OPC_CMPU_LE_QB:

            case OPC_CMP_EQ_PH:

            case OPC_CMP_LT_PH:

            case OPC_CMP_LE_PH:

                gen_mipsdsp_add_cmp_pick(ctx, op1, op2, rd, rs, rt, 0);

                break;

            case OPC_CMPGU_EQ_QB:

            case OPC_CMPGU_LT_QB:

            case OPC_CMPGU_LE_QB:

            case OPC_CMPGDU_EQ_QB:

            case OPC_CMPGDU_LT_QB:

            case OPC_CMPGDU_LE_QB:

            case OPC_PICK_QB:

            case OPC_PICK_PH:

            case OPC_PACKRL_PH:

                gen_mipsdsp_add_cmp_pick(ctx, op1, op2, rd, rs, rt, 1);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK CMPU.EQ.QB");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_SHLL_QB_DSP:

            gen_mipsdsp_shift(ctx, op1, rd, rs, rt);

            break;

        case OPC_DPA_W_PH_DSP:

            op2 = MASK_DPA_W_PH(ctx->opcode);

            switch (op2) {

            case OPC_DPAU_H_QBL:

            case OPC_DPAU_H_QBR:

            case OPC_DPSU_H_QBL:

            case OPC_DPSU_H_QBR:

            case OPC_DPA_W_PH:

            case OPC_DPAX_W_PH:

            case OPC_DPAQ_S_W_PH:

            case OPC_DPAQX_S_W_PH:

            case OPC_DPAQX_SA_W_PH:

            case OPC_DPS_W_PH:

            case OPC_DPSX_W_PH:

            case OPC_DPSQ_S_W_PH:

            case OPC_DPSQX_S_W_PH:

            case OPC_DPSQX_SA_W_PH:

            case OPC_MULSAQ_S_W_PH:

            case OPC_DPAQ_SA_L_W:

            case OPC_DPSQ_SA_L_W:

            case OPC_MAQ_S_W_PHL:

            case OPC_MAQ_S_W_PHR:

            case OPC_MAQ_SA_W_PHL:

            case OPC_MAQ_SA_W_PHR:

            case OPC_MULSA_W_PH:

                gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 0);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK DPAW.PH");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_INSV_DSP:

            op2 = MASK_INSV(ctx->opcode);

            switch (op2) {

            case OPC_INSV:

                check_dsp(ctx);

                {

                    TCGv t0, t1;



                    if (rt == 0) {

                        MIPS_DEBUG("NOP");

                        break;

                    }



                    t0 = tcg_temp_new();

                    t1 = tcg_temp_new();



                    gen_load_gpr(t0, rt);

                    gen_load_gpr(t1, rs);



                    gen_helper_insv(cpu_gpr[rt], cpu_env, t1, t0);



                    tcg_temp_free(t0);

                    tcg_temp_free(t1);

                    break;

                }

            default:            /* Invalid */

                MIPS_INVAL("MASK INSV");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_APPEND_DSP:

            gen_mipsdsp_append(env, ctx, op1, rt, rs, rd);

            break;

        case OPC_EXTR_W_DSP:

            op2 = MASK_EXTR_W(ctx->opcode);

            switch (op2) {

            case OPC_EXTR_W:

            case OPC_EXTR_R_W:

            case OPC_EXTR_RS_W:

            case OPC_EXTR_S_H:

            case OPC_EXTRV_S_H:

            case OPC_EXTRV_W:

            case OPC_EXTRV_R_W:

            case OPC_EXTRV_RS_W:

            case OPC_EXTP:

            case OPC_EXTPV:

            case OPC_EXTPDP:

            case OPC_EXTPDPV:

                gen_mipsdsp_accinsn(ctx, op1, op2, rt, rs, rd, 1);

                break;

            case OPC_RDDSP:

                gen_mipsdsp_accinsn(ctx, op1, op2, rd, rs, rt, 1);

                break;

            case OPC_SHILO:

            case OPC_SHILOV:

            case OPC_MTHLIP:

            case OPC_WRDSP:

                gen_mipsdsp_accinsn(ctx, op1, op2, rd, rs, rt, 0);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK EXTR.W");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

#if defined(TARGET_MIPS64)

        case OPC_DEXTM ... OPC_DEXT:

        case OPC_DINSM ... OPC_DINS:

            check_insn(ctx, ISA_MIPS64R2);

            check_mips_64(ctx);

            gen_bitops(ctx, op1, rt, rs, sa, rd);

            break;

        case OPC_DBSHFL:

            check_insn(ctx, ISA_MIPS64R2);

            check_mips_64(ctx);

            op2 = MASK_DBSHFL(ctx->opcode);

            gen_bshfl(ctx, op2, rt, rd);

            break;

        case OPC_DDIV_G_2E ... OPC_DDIVU_G_2E:

        case OPC_DMULT_G_2E ... OPC_DMULTU_G_2E:

        case OPC_DMOD_G_2E ... OPC_DMODU_G_2E:

            check_insn(ctx, INSN_LOONGSON2E);

            gen_loongson_integer(ctx, op1, rd, rs, rt);

            break;

        case OPC_ABSQ_S_QH_DSP:

            op2 = MASK_ABSQ_S_QH(ctx->opcode);

            switch (op2) {

            case OPC_PRECEQ_L_PWL:

            case OPC_PRECEQ_L_PWR:

            case OPC_PRECEQ_PW_QHL:

            case OPC_PRECEQ_PW_QHR:

            case OPC_PRECEQ_PW_QHLA:

            case OPC_PRECEQ_PW_QHRA:

            case OPC_PRECEQU_QH_OBL:

            case OPC_PRECEQU_QH_OBR:

            case OPC_PRECEQU_QH_OBLA:

            case OPC_PRECEQU_QH_OBRA:

            case OPC_PRECEU_QH_OBL:

            case OPC_PRECEU_QH_OBR:

            case OPC_PRECEU_QH_OBLA:

            case OPC_PRECEU_QH_OBRA:

            case OPC_ABSQ_S_OB:

            case OPC_ABSQ_S_PW:

            case OPC_ABSQ_S_QH:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_REPL_OB:

            case OPC_REPL_PW:

            case OPC_REPL_QH:

            case OPC_REPLV_OB:

            case OPC_REPLV_PW:

            case OPC_REPLV_QH:

                gen_mipsdsp_bitinsn(ctx, op1, op2, rd, rt);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK ABSQ_S.QH");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_ADDU_OB_DSP:

            op2 = MASK_ADDU_OB(ctx->opcode);

            switch (op2) {

            case OPC_RADDU_L_OB:

            case OPC_SUBQ_PW:

            case OPC_SUBQ_S_PW:

            case OPC_SUBQ_QH:

            case OPC_SUBQ_S_QH:

            case OPC_SUBU_OB:

            case OPC_SUBU_S_OB:

            case OPC_SUBU_QH:

            case OPC_SUBU_S_QH:

            case OPC_SUBUH_OB:

            case OPC_SUBUH_R_OB:

            case OPC_ADDQ_PW:

            case OPC_ADDQ_S_PW:

            case OPC_ADDQ_QH:

            case OPC_ADDQ_S_QH:

            case OPC_ADDU_OB:

            case OPC_ADDU_S_OB:

            case OPC_ADDU_QH:

            case OPC_ADDU_S_QH:

            case OPC_ADDUH_OB:

            case OPC_ADDUH_R_OB:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_MULEQ_S_PW_QHL:

            case OPC_MULEQ_S_PW_QHR:

            case OPC_MULEU_S_QH_OBL:

            case OPC_MULEU_S_QH_OBR:

            case OPC_MULQ_RS_QH:

                gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 1);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK ADDU.OB");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_CMPU_EQ_OB_DSP:

            op2 = MASK_CMPU_EQ_OB(ctx->opcode);

            switch (op2) {

            case OPC_PRECR_SRA_QH_PW:

            case OPC_PRECR_SRA_R_QH_PW:

                /* Return value is rt. */

                gen_mipsdsp_arith(ctx, op1, op2, rt, rs, rd);

                break;

            case OPC_PRECR_OB_QH:

            case OPC_PRECRQ_OB_QH:

            case OPC_PRECRQ_PW_L:

            case OPC_PRECRQ_QH_PW:

            case OPC_PRECRQ_RS_QH_PW:

            case OPC_PRECRQU_S_OB_QH:

                gen_mipsdsp_arith(ctx, op1, op2, rd, rs, rt);

                break;

            case OPC_CMPU_EQ_OB:

            case OPC_CMPU_LT_OB:

            case OPC_CMPU_LE_OB:

            case OPC_CMP_EQ_QH:

            case OPC_CMP_LT_QH:

            case OPC_CMP_LE_QH:

            case OPC_CMP_EQ_PW:

            case OPC_CMP_LT_PW:

            case OPC_CMP_LE_PW:

                gen_mipsdsp_add_cmp_pick(ctx, op1, op2, rd, rs, rt, 0);

                break;

            case OPC_CMPGDU_EQ_OB:

            case OPC_CMPGDU_LT_OB:

            case OPC_CMPGDU_LE_OB:

            case OPC_CMPGU_EQ_OB:

            case OPC_CMPGU_LT_OB:

            case OPC_CMPGU_LE_OB:

            case OPC_PACKRL_PW:

            case OPC_PICK_OB:

            case OPC_PICK_PW:

            case OPC_PICK_QH:

                gen_mipsdsp_add_cmp_pick(ctx, op1, op2, rd, rs, rt, 1);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK CMPU_EQ.OB");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DAPPEND_DSP:

            gen_mipsdsp_append(env, ctx, op1, rt, rs, rd);

            break;

        case OPC_DEXTR_W_DSP:

            op2 = MASK_DEXTR_W(ctx->opcode);

            switch (op2) {

            case OPC_DEXTP:

            case OPC_DEXTPDP:

            case OPC_DEXTPDPV:

            case OPC_DEXTPV:

            case OPC_DEXTR_L:

            case OPC_DEXTR_R_L:

            case OPC_DEXTR_RS_L:

            case OPC_DEXTR_W:

            case OPC_DEXTR_R_W:

            case OPC_DEXTR_RS_W:

            case OPC_DEXTR_S_H:

            case OPC_DEXTRV_L:

            case OPC_DEXTRV_R_L:

            case OPC_DEXTRV_RS_L:

            case OPC_DEXTRV_S_H:

            case OPC_DEXTRV_W:

            case OPC_DEXTRV_R_W:

            case OPC_DEXTRV_RS_W:

                gen_mipsdsp_accinsn(ctx, op1, op2, rt, rs, rd, 1);

                break;

            case OPC_DMTHLIP:

            case OPC_DSHILO:

            case OPC_DSHILOV:

                gen_mipsdsp_accinsn(ctx, op1, op2, rd, rs, rt, 0);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK EXTR.W");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DPAQ_W_QH_DSP:

            op2 = MASK_DPAQ_W_QH(ctx->opcode);

            switch (op2) {

            case OPC_DPAU_H_OBL:

            case OPC_DPAU_H_OBR:

            case OPC_DPSU_H_OBL:

            case OPC_DPSU_H_OBR:

            case OPC_DPA_W_QH:

            case OPC_DPAQ_S_W_QH:

            case OPC_DPS_W_QH:

            case OPC_DPSQ_S_W_QH:

            case OPC_MULSAQ_S_W_QH:

            case OPC_DPAQ_SA_L_PW:

            case OPC_DPSQ_SA_L_PW:

            case OPC_MULSAQ_S_L_PW:

                gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 0);

                break;

            case OPC_MAQ_S_W_QHLL:

            case OPC_MAQ_S_W_QHLR:

            case OPC_MAQ_S_W_QHRL:

            case OPC_MAQ_S_W_QHRR:

            case OPC_MAQ_SA_W_QHLL:

            case OPC_MAQ_SA_W_QHLR:

            case OPC_MAQ_SA_W_QHRL:

            case OPC_MAQ_SA_W_QHRR:

            case OPC_MAQ_S_L_PWL:

            case OPC_MAQ_S_L_PWR:

            case OPC_DMADD:

            case OPC_DMADDU:

            case OPC_DMSUB:

            case OPC_DMSUBU:

                gen_mipsdsp_multiply(ctx, op1, op2, rd, rs, rt, 0);

                break;

            default:            /* Invalid */

                MIPS_INVAL("MASK DPAQ.W.QH");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_DINSV_DSP:

            op2 = MASK_INSV(ctx->opcode);

            switch (op2) {

            case OPC_DINSV:

                {

                    TCGv t0, t1;



                    if (rt == 0) {

                        MIPS_DEBUG("NOP");

                        break;

                    }

                    check_dsp(ctx);



                    t0 = tcg_temp_new();

                    t1 = tcg_temp_new();



                    gen_load_gpr(t0, rt);

                    gen_load_gpr(t1, rs);



                    gen_helper_dinsv(cpu_gpr[rt], cpu_env, t1, t0);

                    break;

                }

            default:            /* Invalid */

                MIPS_INVAL("MASK DINSV");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case OPC_SHLL_OB_DSP:

            gen_mipsdsp_shift(ctx, op1, rd, rs, rt);

            break;

#endif

        default:            /* Invalid */

            MIPS_INVAL("special3");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_REGIMM:

        op1 = MASK_REGIMM(ctx->opcode);

        switch (op1) {

        case OPC_BLTZ ... OPC_BGEZL: /* REGIMM branches */

        case OPC_BLTZAL ... OPC_BGEZALL:

            gen_compute_branch(ctx, op1, 4, rs, -1, imm << 2);

            *is_branch = 1;

            break;

        case OPC_TGEI ... OPC_TEQI: /* REGIMM traps */

        case OPC_TNEI:

            gen_trap(ctx, op1, rs, -1, imm);

            break;

        case OPC_SYNCI:

            check_insn(ctx, ISA_MIPS32R2);

            /* Treat as NOP. */

            break;

        case OPC_BPOSGE32:    /* MIPS DSP branch */

#if defined(TARGET_MIPS64)

        case OPC_BPOSGE64:

#endif

            check_dsp(ctx);

            gen_compute_branch(ctx, op1, 4, -1, -2, (int32_t)imm << 2);

            *is_branch = 1;

            break;

        default:            /* Invalid */

            MIPS_INVAL("regimm");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_CP0:

        check_cp0_enabled(ctx);

        op1 = MASK_CP0(ctx->opcode);

        switch (op1) {

        case OPC_MFC0:

        case OPC_MTC0:

        case OPC_MFTR:

        case OPC_MTTR:

#if defined(TARGET_MIPS64)

        case OPC_DMFC0:

        case OPC_DMTC0:

#endif

#ifndef CONFIG_USER_ONLY

            gen_cp0(env, ctx, op1, rt, rd);

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_C0_FIRST ... OPC_C0_LAST:

#ifndef CONFIG_USER_ONLY

            gen_cp0(env, ctx, MASK_C0(ctx->opcode), rt, rd);

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_MFMC0:

#ifndef CONFIG_USER_ONLY

            {

                TCGv t0 = tcg_temp_new();



                op2 = MASK_MFMC0(ctx->opcode);

                switch (op2) {

                case OPC_DMT:

                    check_insn(ctx, ASE_MT);

                    gen_helper_dmt(t0);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_EMT:

                    check_insn(ctx, ASE_MT);

                    gen_helper_emt(t0);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_DVPE:

                    check_insn(ctx, ASE_MT);

                    gen_helper_dvpe(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_EVPE:

                    check_insn(ctx, ASE_MT);

                    gen_helper_evpe(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_DI:

                    check_insn(ctx, ISA_MIPS32R2);

                    save_cpu_state(ctx, 1);

                    gen_helper_di(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    /* Stop translation as we may have switched the execution mode */

                    ctx->bstate = BS_STOP;

                    break;

                case OPC_EI:

                    check_insn(ctx, ISA_MIPS32R2);

                    save_cpu_state(ctx, 1);

                    gen_helper_ei(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    /* Stop translation as we may have switched the execution mode */

                    ctx->bstate = BS_STOP;

                    break;

                default:            /* Invalid */

                    MIPS_INVAL("mfmc0");

                    generate_exception(ctx, EXCP_RI);

                    break;

                }

                tcg_temp_free(t0);

            }

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_RDPGPR:

            check_insn(ctx, ISA_MIPS32R2);

            gen_load_srsgpr(rt, rd);

            break;

        case OPC_WRPGPR:

            check_insn(ctx, ISA_MIPS32R2);

            gen_store_srsgpr(rt, rd);

            break;

        default:

            MIPS_INVAL("cp0");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_ADDI: /* Arithmetic with immediate opcode */

    case OPC_ADDIU:

         gen_arith_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_SLTI: /* Set on less than with immediate opcode */

    case OPC_SLTIU:

         gen_slt_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_ANDI: /* Arithmetic with immediate opcode */

    case OPC_LUI:

    case OPC_ORI:

    case OPC_XORI:

         gen_logic_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_J ... OPC_JAL: /* Jump */

         offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

         gen_compute_branch(ctx, op, 4, rs, rt, offset);

         *is_branch = 1;

         break;

    case OPC_BEQ ... OPC_BGTZ: /* Branch */

    case OPC_BEQL ... OPC_BGTZL:

         gen_compute_branch(ctx, op, 4, rs, rt, imm << 2);

         *is_branch = 1;

         break;

    case OPC_LB ... OPC_LWR: /* Load and stores */

    case OPC_LL:

         gen_ld(ctx, op, rt, rs, imm);

         break;

    case OPC_SB ... OPC_SW:

    case OPC_SWR:

         gen_st(ctx, op, rt, rs, imm);

         break;

    case OPC_SC:

         gen_st_cond(ctx, op, rt, rs, imm);

         break;

    case OPC_CACHE:

        check_cp0_enabled(ctx);

        check_insn(ctx, ISA_MIPS3 | ISA_MIPS32);

        /* Treat as NOP. */

        break;

    case OPC_PREF:

        check_insn(ctx, ISA_MIPS4 | ISA_MIPS32);

        /* Treat as NOP. */

        break;



    /* Floating point (COP1). */

    case OPC_LWC1:

    case OPC_LDC1:

    case OPC_SWC1:

    case OPC_SDC1:

        gen_cop1_ldst(env, ctx, op, rt, rs, imm);

        break;



    case OPC_CP1:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            check_cp1_enabled(ctx);

            op1 = MASK_CP1(ctx->opcode);

            switch (op1) {

            case OPC_MFHC1:

            case OPC_MTHC1:

                check_insn(ctx, ISA_MIPS32R2);

            case OPC_MFC1:

            case OPC_CFC1:

            case OPC_MTC1:

            case OPC_CTC1:

                gen_cp1(ctx, op1, rt, rd);

                break;

#if defined(TARGET_MIPS64)

            case OPC_DMFC1:

            case OPC_DMTC1:

                check_insn(ctx, ISA_MIPS3);

                gen_cp1(ctx, op1, rt, rd);

                break;

#endif

            case OPC_BC1ANY2:

            case OPC_BC1ANY4:

                check_cop1x(ctx);

                check_insn(ctx, ASE_MIPS3D);

                /* fall through */

            case OPC_BC1:

                gen_compute_branch1(ctx, MASK_BC1(ctx->opcode),

                                    (rt >> 2) & 0x7, imm << 2);

                *is_branch = 1;

                break;

            case OPC_S_FMT:

            case OPC_D_FMT:

            case OPC_W_FMT:

            case OPC_L_FMT:

            case OPC_PS_FMT:

                gen_farith(ctx, ctx->opcode & FOP(0x3f, 0x1f), rt, rd, sa,

                           (imm >> 8) & 0x7);

                break;

            default:

                MIPS_INVAL("cp1");

                generate_exception (ctx, EXCP_RI);

                break;

            }

        } else {

            generate_exception_err(ctx, EXCP_CpU, 1);

        }

        break;



    /* COP2.  */

    case OPC_LWC2:

    case OPC_LDC2:

    case OPC_SWC2:

    case OPC_SDC2:

        /* COP2: Not implemented. */

        generate_exception_err(ctx, EXCP_CpU, 2);

        break;

    case OPC_CP2:

        check_insn(ctx, INSN_LOONGSON2F);

        /* Note that these instructions use different fields.  */

        gen_loongson_multimedia(ctx, sa, rd, rt);

        break;



    case OPC_CP3:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            check_cp1_enabled(ctx);

            op1 = MASK_CP3(ctx->opcode);

            switch (op1) {

            case OPC_LWXC1:

            case OPC_LDXC1:

            case OPC_LUXC1:

            case OPC_SWXC1:

            case OPC_SDXC1:

            case OPC_SUXC1:

                gen_flt3_ldst(ctx, op1, sa, rd, rs, rt);

                break;

            case OPC_PREFX:

                /* Treat as NOP. */

                break;

            case OPC_ALNV_PS:

            case OPC_MADD_S:

            case OPC_MADD_D:

            case OPC_MADD_PS:

            case OPC_MSUB_S:

            case OPC_MSUB_D:

            case OPC_MSUB_PS:

            case OPC_NMADD_S:

            case OPC_NMADD_D:

            case OPC_NMADD_PS:

            case OPC_NMSUB_S:

            case OPC_NMSUB_D:

            case OPC_NMSUB_PS:

                gen_flt3_arith(ctx, op1, sa, rs, rd, rt);

                break;

            default:

                MIPS_INVAL("cp3");

                generate_exception (ctx, EXCP_RI);

                break;

            }

        } else {

            generate_exception_err(ctx, EXCP_CpU, 1);

        }

        break;



#if defined(TARGET_MIPS64)

    /* MIPS64 opcodes */

    case OPC_LWU:

    case OPC_LDL ... OPC_LDR:

    case OPC_LLD:

    case OPC_LD:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_ld(ctx, op, rt, rs, imm);

        break;

    case OPC_SDL ... OPC_SDR:

    case OPC_SD:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_st(ctx, op, rt, rs, imm);

        break;

    case OPC_SCD:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_st_cond(ctx, op, rt, rs, imm);

        break;

    case OPC_DADDI:

    case OPC_DADDIU:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_arith_imm(ctx, op, rt, rs, imm);

        break;

#endif

    case OPC_JALX:

        check_insn(ctx, ASE_MIPS16 | ASE_MICROMIPS);

        offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

        gen_compute_branch(ctx, op, 4, rs, rt, offset);

        *is_branch = 1;

        break;

    case OPC_MDMX:

        check_insn(ctx, ASE_MDMX);

        /* MDMX: Not implemented. */

    default:            /* Invalid */

        MIPS_INVAL("major opcode");

        generate_exception(ctx, EXCP_RI);

        break;

    }

}
