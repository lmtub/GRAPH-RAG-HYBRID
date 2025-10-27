static void decode_opc (CPUState *env, DisasContext *ctx)

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



    if ((ctx->hflags & MIPS_HFLAG_BMASK) == MIPS_HFLAG_BL) {

        int l1;

        /* Handle blikely not taken case */

        MIPS_DEBUG("blikely condition (" TARGET_FMT_lx ")", ctx->pc + 4);

        l1 = gen_new_label();

        gen_op_jnz_T2(l1);

        gen_op_save_state(ctx->hflags & ~MIPS_HFLAG_BMASK);

        gen_goto_tb(ctx, 1, ctx->pc + 4);

        gen_set_label(l1);

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

        case OPC_SLL:          /* Arithmetic with immediate */

        case OPC_SRL ... OPC_SRA:

            gen_arith_imm(env, ctx, op1, rd, rt, sa);

            break;

        case OPC_MOVZ ... OPC_MOVN:

            check_insn(env, ctx, ISA_MIPS4 | ISA_MIPS32);

        case OPC_SLLV:         /* Arithmetic */

        case OPC_SRLV ... OPC_SRAV:

        case OPC_ADD ... OPC_NOR:

        case OPC_SLT ... OPC_SLTU:

            gen_arith(env, ctx, op1, rd, rs, rt);

            break;

        case OPC_MULT ... OPC_DIVU:

            gen_muldiv(ctx, op1, rs, rt);

            break;

        case OPC_JR ... OPC_JALR:

            gen_compute_branch(ctx, op1, rs, rd, sa);

            return;

        case OPC_TGE ... OPC_TEQ: /* Traps */

        case OPC_TNE:

            gen_trap(ctx, op1, rs, rt, -1);

            break;

        case OPC_MFHI:          /* Move from HI/LO */

        case OPC_MFLO:

            gen_HILO(ctx, op1, rd);

            break;

        case OPC_MTHI:

        case OPC_MTLO:          /* Move to HI/LO */

            gen_HILO(ctx, op1, rs);

            break;

        case OPC_PMON:          /* Pmon entry point, also R4010 selsl */

#ifdef MIPS_STRICT_STANDARD

            MIPS_INVAL("PMON / selsl");

            generate_exception(ctx, EXCP_RI);

#else

            gen_op_pmon(sa);

#endif

            break;

        case OPC_SYSCALL:

            generate_exception(ctx, EXCP_SYSCALL);

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

            check_insn(env, ctx, ISA_MIPS4 | ISA_MIPS32);

            if (env->CP0_Config1 & (1 << CP0C1_FP)) {

                save_cpu_state(ctx, 1);

                check_cp1_enabled(ctx);

                gen_movci(ctx, rd, rs, (ctx->opcode >> 18) & 0x7,

                          (ctx->opcode >> 16) & 1);

            } else {

                generate_exception_err(ctx, EXCP_CpU, 1);

            }

            break;



#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

       /* MIPS64 specific opcodes */

        case OPC_DSLL:

        case OPC_DSRL ... OPC_DSRA:

        case OPC_DSLL32:

        case OPC_DSRL32 ... OPC_DSRA32:

            check_insn(env, ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_arith_imm(env, ctx, op1, rd, rt, sa);

            break;

        case OPC_DSLLV:

        case OPC_DSRLV ... OPC_DSRAV:

        case OPC_DADD ... OPC_DSUBU:

            check_insn(env, ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_arith(env, ctx, op1, rd, rs, rt);

            break;

        case OPC_DMULT ... OPC_DDIVU:

            check_insn(env, ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_muldiv(ctx, op1, rs, rt);

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

            check_insn(env, ctx, ISA_MIPS32);

            gen_muldiv(ctx, op1, rs, rt);

            break;

        case OPC_MUL:

            gen_arith(env, ctx, op1, rd, rs, rt);

            break;

        case OPC_CLZ ... OPC_CLO:

            check_insn(env, ctx, ISA_MIPS32);

            gen_cl(ctx, op1, rd, rs);

            break;

        case OPC_SDBBP:

            /* XXX: not clear which exception should be raised

             *      when in debug mode...

             */

            check_insn(env, ctx, ISA_MIPS32);

            if (!(ctx->hflags & MIPS_HFLAG_DM)) {

                generate_exception(ctx, EXCP_DBp);

            } else {

                generate_exception(ctx, EXCP_DBp);

            }

            /* Treat as NOP. */

            break;

#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

        case OPC_DCLZ ... OPC_DCLO:

            check_insn(env, ctx, ISA_MIPS64);

            check_mips_64(ctx);

            gen_cl(ctx, op1, rd, rs);

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

             check_insn(env, ctx, ISA_MIPS32R2);

             gen_bitops(ctx, op1, rt, rs, sa, rd);

             break;

         case OPC_BSHFL:

             check_insn(env, ctx, ISA_MIPS32R2);

             op2 = MASK_BSHFL(ctx->opcode);

             switch (op2) {

             case OPC_WSBH:

                 GEN_LOAD_REG_TN(T1, rt);

                 gen_op_wsbh();

                 break;

             case OPC_SEB:

                 GEN_LOAD_REG_TN(T1, rt);

                 gen_op_seb();

                 break;

             case OPC_SEH:

                 GEN_LOAD_REG_TN(T1, rt);

                 gen_op_seh();

                 break;

             default:            /* Invalid */

                 MIPS_INVAL("bshfl");

                 generate_exception(ctx, EXCP_RI);

                 break;

            }

            GEN_STORE_TN_REG(rd, T0);

            break;

        case OPC_RDHWR:

            check_insn(env, ctx, ISA_MIPS32R2);

            switch (rd) {

            case 0:

                save_cpu_state(ctx, 1);

                gen_op_rdhwr_cpunum();

                break;

            case 1:

                save_cpu_state(ctx, 1);

                gen_op_rdhwr_synci_step();

                break;

            case 2:

                save_cpu_state(ctx, 1);

                gen_op_rdhwr_cc();

                break;

            case 3:

                save_cpu_state(ctx, 1);

                gen_op_rdhwr_ccres();

                break;

            case 29:

#if defined (CONFIG_USER_ONLY)

                gen_op_tls_value();

                break;

#endif

            default:            /* Invalid */

                MIPS_INVAL("rdhwr");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            GEN_STORE_TN_REG(rt, T0);

            break;

        case OPC_FORK:

            check_mips_mt(env, ctx);

            GEN_LOAD_REG_TN(T0, rt);

            GEN_LOAD_REG_TN(T1, rs);

            gen_op_fork();

            break;

        case OPC_YIELD:

            check_mips_mt(env, ctx);

            GEN_LOAD_REG_TN(T0, rs);

            gen_op_yield();

            GEN_STORE_TN_REG(rd, T0);

            break;

#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

        case OPC_DEXTM ... OPC_DEXT:

        case OPC_DINSM ... OPC_DINS:

            check_insn(env, ctx, ISA_MIPS64R2);

            check_mips_64(ctx);

            gen_bitops(ctx, op1, rt, rs, sa, rd);

            break;

        case OPC_DBSHFL:

            check_insn(env, ctx, ISA_MIPS64R2);

            check_mips_64(ctx);

            op2 = MASK_DBSHFL(ctx->opcode);

            switch (op2) {

            case OPC_DSBH:

                GEN_LOAD_REG_TN(T1, rt);

                gen_op_dsbh();

                break;

            case OPC_DSHD:

                GEN_LOAD_REG_TN(T1, rt);

                gen_op_dshd();

                break;

            default:            /* Invalid */

                MIPS_INVAL("dbshfl");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            GEN_STORE_TN_REG(rd, T0);

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

            gen_compute_branch(ctx, op1, rs, -1, imm << 2);

            return;

        case OPC_TGEI ... OPC_TEQI: /* REGIMM traps */

        case OPC_TNEI:

            gen_trap(ctx, op1, rs, -1, imm);

            break;

        case OPC_SYNCI:

            check_insn(env, ctx, ISA_MIPS32R2);

            /* Treat as NOP. */

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

#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

        case OPC_DMFC0:

        case OPC_DMTC0:

#endif

            gen_cp0(env, ctx, op1, rt, rd);

            break;

        case OPC_C0_FIRST ... OPC_C0_LAST:

            gen_cp0(env, ctx, MASK_C0(ctx->opcode), rt, rd);

            break;

        case OPC_MFMC0:

            op2 = MASK_MFMC0(ctx->opcode);

            switch (op2) {

            case OPC_DMT:

                check_mips_mt(env, ctx);

                gen_op_dmt();

                break;

            case OPC_EMT:

                check_mips_mt(env, ctx);

                gen_op_emt();

                break;

            case OPC_DVPE:

                check_mips_mt(env, ctx);

                gen_op_dvpe();

                break;

            case OPC_EVPE:

                check_mips_mt(env, ctx);

                gen_op_evpe();

                break;

            case OPC_DI:

                check_insn(env, ctx, ISA_MIPS32R2);

                save_cpu_state(ctx, 1);

                gen_op_di();

                /* Stop translation as we may have switched the execution mode */

                ctx->bstate = BS_STOP;

                break;

            case OPC_EI:

                check_insn(env, ctx, ISA_MIPS32R2);

                save_cpu_state(ctx, 1);

                gen_op_ei();

                /* Stop translation as we may have switched the execution mode */

                ctx->bstate = BS_STOP;

                break;

            default:            /* Invalid */

                MIPS_INVAL("mfmc0");

                generate_exception(ctx, EXCP_RI);

                break;

            }

            GEN_STORE_TN_REG(rt, T0);

            break;

        case OPC_RDPGPR:

            check_insn(env, ctx, ISA_MIPS32R2);

            GEN_LOAD_SRSREG_TN(T0, rt);

            GEN_STORE_TN_REG(rd, T0);

            break;

        case OPC_WRPGPR:

            check_insn(env, ctx, ISA_MIPS32R2);

            GEN_LOAD_REG_TN(T0, rt);

            GEN_STORE_TN_SRSREG(rd, T0);

            break;

        default:

            MIPS_INVAL("cp0");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_ADDI ... OPC_LUI: /* Arithmetic with immediate opcode */

         gen_arith_imm(env, ctx, op, rt, rs, imm);

         break;

    case OPC_J ... OPC_JAL: /* Jump */

         offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

         gen_compute_branch(ctx, op, rs, rt, offset);

         return;

    case OPC_BEQ ... OPC_BGTZ: /* Branch */

    case OPC_BEQL ... OPC_BGTZL:

         gen_compute_branch(ctx, op, rs, rt, imm << 2);

         return;

    case OPC_LB ... OPC_LWR: /* Load and stores */

    case OPC_SB ... OPC_SW:

    case OPC_SWR:

    case OPC_LL:

    case OPC_SC:

         gen_ldst(ctx, op, rt, rs, imm);

         break;

    case OPC_CACHE:

        check_insn(env, ctx, ISA_MIPS3 | ISA_MIPS32);

        /* Treat as NOP. */

        break;

    case OPC_PREF:

        check_insn(env, ctx, ISA_MIPS4 | ISA_MIPS32);

        /* Treat as NOP. */

        break;



    /* Floating point (COP1). */

    case OPC_LWC1:

    case OPC_LDC1:

    case OPC_SWC1:

    case OPC_SDC1:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            save_cpu_state(ctx, 1);

            check_cp1_enabled(ctx);

            gen_flt_ldst(ctx, op, rt, rs, imm);

        } else {

            generate_exception_err(ctx, EXCP_CpU, 1);

        }

        break;



    case OPC_CP1:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            save_cpu_state(ctx, 1);

            check_cp1_enabled(ctx);

            op1 = MASK_CP1(ctx->opcode);

            switch (op1) {

            case OPC_MFHC1:

            case OPC_MTHC1:

                check_insn(env, ctx, ISA_MIPS32R2);

            case OPC_MFC1:

            case OPC_CFC1:

            case OPC_MTC1:

            case OPC_CTC1:

                gen_cp1(ctx, op1, rt, rd);

                break;

#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

            case OPC_DMFC1:

            case OPC_DMTC1:

                check_insn(env, ctx, ISA_MIPS3);

                gen_cp1(ctx, op1, rt, rd);

                break;

#endif

            case OPC_BC1ANY2:

            case OPC_BC1ANY4:

                check_cp1_3d(env, ctx);

                /* fall through */

            case OPC_BC1:

                gen_compute_branch1(env, ctx, MASK_BC1(ctx->opcode),

                                    (rt >> 2) & 0x7, imm << 2);

                return;

            case OPC_S_FMT:

            case OPC_D_FMT:

            case OPC_W_FMT:

            case OPC_L_FMT:

            case OPC_PS_FMT:

                gen_farith(ctx, MASK_CP1_FUNC(ctx->opcode), rt, rd, sa,

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

    case OPC_CP2:

        /* COP2: Not implemented. */

        generate_exception_err(ctx, EXCP_CpU, 2);

        break;



    case OPC_CP3:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            save_cpu_state(ctx, 1);

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



#if defined(TARGET_MIPSN32) || defined(TARGET_MIPS64)

    /* MIPS64 opcodes */

    case OPC_LWU:

    case OPC_LDL ... OPC_LDR:

    case OPC_SDL ... OPC_SDR:

    case OPC_LLD:

    case OPC_LD:

    case OPC_SCD:

    case OPC_SD:

        check_insn(env, ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_ldst(ctx, op, rt, rs, imm);

        break;

    case OPC_DADDI ... OPC_DADDIU:

        check_insn(env, ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_arith_imm(env, ctx, op, rt, rs, imm);

        break;

#endif

    case OPC_JALX:

        check_insn(env, ctx, ASE_MIPS16);

        /* MIPS16: Not implemented. */

    case OPC_MDMX:

        check_insn(env, ctx, ASE_MDMX);

        /* MDMX: Not implemented. */

    default:            /* Invalid */

        MIPS_INVAL("major opcode");

        generate_exception(ctx, EXCP_RI);

        break;

    }

    if (ctx->hflags & MIPS_HFLAG_BMASK) {

        int hflags = ctx->hflags & MIPS_HFLAG_BMASK;

        /* Branches completion */

        ctx->hflags &= ~MIPS_HFLAG_BMASK;

        ctx->bstate = BS_BRANCH;

        save_cpu_state(ctx, 0);

        switch (hflags) {

        case MIPS_HFLAG_B:

            /* unconditional branch */

            MIPS_DEBUG("unconditional branch");

            gen_goto_tb(ctx, 0, ctx->btarget);

            break;

        case MIPS_HFLAG_BL:

            /* blikely taken case */

            MIPS_DEBUG("blikely branch taken");

            gen_goto_tb(ctx, 0, ctx->btarget);

            break;

        case MIPS_HFLAG_BC:

            /* Conditional branch */

            MIPS_DEBUG("conditional branch");

            {

              int l1;

              l1 = gen_new_label();

              gen_op_jnz_T2(l1);

              gen_goto_tb(ctx, 1, ctx->pc + 4);

              gen_set_label(l1);

              gen_goto_tb(ctx, 0, ctx->btarget);

            }

            break;

        case MIPS_HFLAG_BR:

            /* unconditional branch to register */

            MIPS_DEBUG("branch to register");

            gen_op_breg();

            gen_op_reset_T0();

            gen_op_exit_tb();

            break;

        default:

            MIPS_DEBUG("unknown branch");

            break;

        }

    }

}
