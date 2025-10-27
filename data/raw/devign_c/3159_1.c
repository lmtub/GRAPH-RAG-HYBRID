static void decode_micromips32_opc (CPUMIPSState *env, DisasContext *ctx,

                                    uint16_t insn_hw1, int *is_branch)

{

    int32_t offset;

    uint16_t insn;

    int rt, rs, rd, rr;

    int16_t imm;

    uint32_t op, minor, mips32_op;

    uint32_t cond, fmt, cc;



    insn = lduw_code(ctx->pc + 2);

    ctx->opcode = (ctx->opcode << 16) | insn;



    rt = (ctx->opcode >> 21) & 0x1f;

    rs = (ctx->opcode >> 16) & 0x1f;

    rd = (ctx->opcode >> 11) & 0x1f;

    rr = (ctx->opcode >> 6) & 0x1f;

    imm = (int16_t) ctx->opcode;



    op = (ctx->opcode >> 26) & 0x3f;

    switch (op) {

    case POOL32A:

        minor = ctx->opcode & 0x3f;

        switch (minor) {

        case 0x00:

            minor = (ctx->opcode >> 6) & 0xf;

            switch (minor) {

            case SLL32:

                mips32_op = OPC_SLL;

                goto do_shifti;

            case SRA:

                mips32_op = OPC_SRA;

                goto do_shifti;

            case SRL32:

                mips32_op = OPC_SRL;

                goto do_shifti;

            case ROTR:

                mips32_op = OPC_ROTR;

            do_shifti:

                gen_shift_imm(env, ctx, mips32_op, rt, rs, rd);

                break;

            default:

                goto pool32a_invalid;

            }

            break;

        case 0x10:

            minor = (ctx->opcode >> 6) & 0xf;

            switch (minor) {

                /* Arithmetic */

            case ADD:

                mips32_op = OPC_ADD;

                goto do_arith;

            case ADDU32:

                mips32_op = OPC_ADDU;

                goto do_arith;

            case SUB:

                mips32_op = OPC_SUB;

                goto do_arith;

            case SUBU32:

                mips32_op = OPC_SUBU;

                goto do_arith;

            case MUL:

                mips32_op = OPC_MUL;

            do_arith:

                gen_arith(env, ctx, mips32_op, rd, rs, rt);

                break;

                /* Shifts */

            case SLLV:

                mips32_op = OPC_SLLV;

                goto do_shift;

            case SRLV:

                mips32_op = OPC_SRLV;

                goto do_shift;

            case SRAV:

                mips32_op = OPC_SRAV;

                goto do_shift;

            case ROTRV:

                mips32_op = OPC_ROTRV;

            do_shift:

                gen_shift(env, ctx, mips32_op, rd, rs, rt);

                break;

                /* Logical operations */

            case AND:

                mips32_op = OPC_AND;

                goto do_logic;

            case OR32:

                mips32_op = OPC_OR;

                goto do_logic;

            case NOR:

                mips32_op = OPC_NOR;

                goto do_logic;

            case XOR32:

                mips32_op = OPC_XOR;

            do_logic:

                gen_logic(env, mips32_op, rd, rs, rt);

                break;

                /* Set less than */

            case SLT:

                mips32_op = OPC_SLT;

                goto do_slt;

            case SLTU:

                mips32_op = OPC_SLTU;

            do_slt:

                gen_slt(env, mips32_op, rd, rs, rt);

                break;

            default:

                goto pool32a_invalid;

            }

            break;

        case 0x18:

            minor = (ctx->opcode >> 6) & 0xf;

            switch (minor) {

                /* Conditional moves */

            case MOVN:

                mips32_op = OPC_MOVN;

                goto do_cmov;

            case MOVZ:

                mips32_op = OPC_MOVZ;

            do_cmov:

                gen_cond_move(env, mips32_op, rd, rs, rt);

                break;

            case LWXS:

                gen_ldxs(ctx, rs, rt, rd);

                break;

            default:

                goto pool32a_invalid;

            }

            break;

        case INS:

            gen_bitops(ctx, OPC_INS, rt, rs, rr, rd);

            return;

        case EXT:

            gen_bitops(ctx, OPC_EXT, rt, rs, rr, rd);

            return;

        case POOL32AXF:

            gen_pool32axf(env, ctx, rt, rs, is_branch);

            break;

        case 0x07:

            generate_exception(ctx, EXCP_BREAK);

            break;

        default:

        pool32a_invalid:

                MIPS_INVAL("pool32a");

                generate_exception(ctx, EXCP_RI);

                break;

        }

        break;

    case POOL32B:

        minor = (ctx->opcode >> 12) & 0xf;

        switch (minor) {

        case CACHE:


            /* Treat as no-op. */

            break;

        case LWC2:

        case SWC2:

            /* COP2: Not implemented. */

            generate_exception_err(ctx, EXCP_CpU, 2);

            break;

        case LWP:

        case SWP:

#ifdef TARGET_MIPS64

        case LDP:

        case SDP:

#endif

            gen_ldst_pair(ctx, minor, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

        case LWM32:

        case SWM32:

#ifdef TARGET_MIPS64

        case LDM:

        case SDM:

#endif

            gen_ldst_multiple(ctx, minor, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

        default:

            MIPS_INVAL("pool32b");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case POOL32F:

        if (env->CP0_Config1 & (1 << CP0C1_FP)) {

            minor = ctx->opcode & 0x3f;

            check_cp1_enabled(ctx);

            switch (minor) {

            case ALNV_PS:

                mips32_op = OPC_ALNV_PS;

                goto do_madd;

            case MADD_S:

                mips32_op = OPC_MADD_S;

                goto do_madd;

            case MADD_D:

                mips32_op = OPC_MADD_D;

                goto do_madd;

            case MADD_PS:

                mips32_op = OPC_MADD_PS;

                goto do_madd;

            case MSUB_S:

                mips32_op = OPC_MSUB_S;

                goto do_madd;

            case MSUB_D:

                mips32_op = OPC_MSUB_D;

                goto do_madd;

            case MSUB_PS:

                mips32_op = OPC_MSUB_PS;

                goto do_madd;

            case NMADD_S:

                mips32_op = OPC_NMADD_S;

                goto do_madd;

            case NMADD_D:

                mips32_op = OPC_NMADD_D;

                goto do_madd;

            case NMADD_PS:

                mips32_op = OPC_NMADD_PS;

                goto do_madd;

            case NMSUB_S:

                mips32_op = OPC_NMSUB_S;

                goto do_madd;

            case NMSUB_D:

                mips32_op = OPC_NMSUB_D;

                goto do_madd;

            case NMSUB_PS:

                mips32_op = OPC_NMSUB_PS;

            do_madd:

                gen_flt3_arith(ctx, mips32_op, rd, rr, rs, rt);

                break;

            case CABS_COND_FMT:

                cond = (ctx->opcode >> 6) & 0xf;

                cc = (ctx->opcode >> 13) & 0x7;

                fmt = (ctx->opcode >> 10) & 0x3;

                switch (fmt) {

                case 0x0:

                    gen_cmpabs_s(ctx, cond, rt, rs, cc);

                    break;

                case 0x1:

                    gen_cmpabs_d(ctx, cond, rt, rs, cc);

                    break;

                case 0x2:

                    gen_cmpabs_ps(ctx, cond, rt, rs, cc);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            case C_COND_FMT:

                cond = (ctx->opcode >> 6) & 0xf;

                cc = (ctx->opcode >> 13) & 0x7;

                fmt = (ctx->opcode >> 10) & 0x3;

                switch (fmt) {

                case 0x0:

                    gen_cmp_s(ctx, cond, rt, rs, cc);

                    break;

                case 0x1:

                    gen_cmp_d(ctx, cond, rt, rs, cc);

                    break;

                case 0x2:

                    gen_cmp_ps(ctx, cond, rt, rs, cc);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            case POOL32FXF:

                gen_pool32fxf(env, ctx, rt, rs);

                break;

            case 0x00:

                /* PLL foo */

                switch ((ctx->opcode >> 6) & 0x7) {

                case PLL_PS:

                    mips32_op = OPC_PLL_PS;

                    goto do_ps;

                case PLU_PS:

                    mips32_op = OPC_PLU_PS;

                    goto do_ps;

                case PUL_PS:

                    mips32_op = OPC_PUL_PS;

                    goto do_ps;

                case PUU_PS:

                    mips32_op = OPC_PUU_PS;

                    goto do_ps;

                case CVT_PS_S:

                    mips32_op = OPC_CVT_PS_S;

                do_ps:

                    gen_farith(ctx, mips32_op, rt, rs, rd, 0);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            case 0x08:

                /* [LS][WDU]XC1 */

                switch ((ctx->opcode >> 6) & 0x7) {

                case LWXC1:

                    mips32_op = OPC_LWXC1;

                    goto do_ldst_cp1;

                case SWXC1:

                    mips32_op = OPC_SWXC1;

                    goto do_ldst_cp1;

                case LDXC1:

                    mips32_op = OPC_LDXC1;

                    goto do_ldst_cp1;

                case SDXC1:

                    mips32_op = OPC_SDXC1;

                    goto do_ldst_cp1;

                case LUXC1:

                    mips32_op = OPC_LUXC1;

                    goto do_ldst_cp1;

                case SUXC1:

                    mips32_op = OPC_SUXC1;

                do_ldst_cp1:

                    gen_flt3_ldst(ctx, mips32_op, rd, rd, rt, rs);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            case 0x18:

                /* 3D insns */

                fmt = (ctx->opcode >> 9) & 0x3;

                switch ((ctx->opcode >> 6) & 0x7) {

                case RSQRT2_FMT:

                    switch (fmt) {

                    case FMT_SDPS_S:

                        mips32_op = OPC_RSQRT2_S;

                        goto do_3d;

                    case FMT_SDPS_D:

                        mips32_op = OPC_RSQRT2_D;

                        goto do_3d;

                    case FMT_SDPS_PS:

                        mips32_op = OPC_RSQRT2_PS;

                        goto do_3d;

                    default:

                        goto pool32f_invalid;

                    }

                    break;

                case RECIP2_FMT:

                    switch (fmt) {

                    case FMT_SDPS_S:

                        mips32_op = OPC_RECIP2_S;

                        goto do_3d;

                    case FMT_SDPS_D:

                        mips32_op = OPC_RECIP2_D;

                        goto do_3d;

                    case FMT_SDPS_PS:

                        mips32_op = OPC_RECIP2_PS;

                        goto do_3d;

                    default:

                        goto pool32f_invalid;

                    }

                    break;

                case ADDR_PS:

                    mips32_op = OPC_ADDR_PS;

                    goto do_3d;

                case MULR_PS:

                    mips32_op = OPC_MULR_PS;

                do_3d:

                    gen_farith(ctx, mips32_op, rt, rs, rd, 0);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            case 0x20:

                /* MOV[FT].fmt and PREFX */

                cc = (ctx->opcode >> 13) & 0x7;

                fmt = (ctx->opcode >> 9) & 0x3;

                switch ((ctx->opcode >> 6) & 0x7) {

                case MOVF_FMT:

                    switch (fmt) {

                    case FMT_SDPS_S:

                        gen_movcf_s(rs, rt, cc, 0);

                        break;

                    case FMT_SDPS_D:

                        gen_movcf_d(ctx, rs, rt, cc, 0);

                        break;

                    case FMT_SDPS_PS:

                        gen_movcf_ps(rs, rt, cc, 0);

                        break;

                    default:

                        goto pool32f_invalid;

                    }

                    break;

                case MOVT_FMT:

                    switch (fmt) {

                    case FMT_SDPS_S:

                        gen_movcf_s(rs, rt, cc, 1);

                        break;

                    case FMT_SDPS_D:

                        gen_movcf_d(ctx, rs, rt, cc, 1);

                        break;

                    case FMT_SDPS_PS:

                        gen_movcf_ps(rs, rt, cc, 1);

                        break;

                    default:

                        goto pool32f_invalid;

                    }

                    break;

                case PREFX:

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

#define FINSN_3ARG_SDPS(prfx)                           \

                switch ((ctx->opcode >> 8) & 0x3) {     \

                case FMT_SDPS_S:                        \

                    mips32_op = OPC_##prfx##_S;         \

                    goto do_fpop;                       \

                case FMT_SDPS_D:                        \

                    mips32_op = OPC_##prfx##_D;         \

                    goto do_fpop;                       \

                case FMT_SDPS_PS:                       \

                    mips32_op = OPC_##prfx##_PS;        \

                    goto do_fpop;                       \

                default:                                \

                    goto pool32f_invalid;               \

                }

            case 0x30:

                /* regular FP ops */

                switch ((ctx->opcode >> 6) & 0x3) {

                case ADD_FMT:

                    FINSN_3ARG_SDPS(ADD);

                    break;

                case SUB_FMT:

                    FINSN_3ARG_SDPS(SUB);

                    break;

                case MUL_FMT:

                    FINSN_3ARG_SDPS(MUL);

                    break;

                case DIV_FMT:

                    fmt = (ctx->opcode >> 8) & 0x3;

                    if (fmt == 1) {

                        mips32_op = OPC_DIV_D;

                    } else if (fmt == 0) {

                        mips32_op = OPC_DIV_S;

                    } else {

                        goto pool32f_invalid;

                    }

                    goto do_fpop;

                default:

                    goto pool32f_invalid;

                }

                break;

            case 0x38:

                /* cmovs */

                switch ((ctx->opcode >> 6) & 0x3) {

                case MOVN_FMT:

                    FINSN_3ARG_SDPS(MOVN);

                    break;

                case MOVZ_FMT:

                    FINSN_3ARG_SDPS(MOVZ);

                    break;

                default:

                    goto pool32f_invalid;

                }

                break;

            do_fpop:

                gen_farith(ctx, mips32_op, rt, rs, rd, 0);

                break;

            default:

            pool32f_invalid:

                MIPS_INVAL("pool32f");

                generate_exception(ctx, EXCP_RI);

                break;

            }

        } else {

            generate_exception_err(ctx, EXCP_CpU, 1);

        }

        break;

    case POOL32I:

        minor = (ctx->opcode >> 21) & 0x1f;

        switch (minor) {

        case BLTZ:

            mips32_op = OPC_BLTZ;

            goto do_branch;

        case BLTZAL:

            mips32_op = OPC_BLTZAL;

            goto do_branch;

        case BLTZALS:

            mips32_op = OPC_BLTZALS;

            goto do_branch;

        case BGEZ:

            mips32_op = OPC_BGEZ;

            goto do_branch;

        case BGEZAL:

            mips32_op = OPC_BGEZAL;

            goto do_branch;

        case BGEZALS:

            mips32_op = OPC_BGEZALS;

            goto do_branch;

        case BLEZ:

            mips32_op = OPC_BLEZ;

            goto do_branch;

        case BGTZ:

            mips32_op = OPC_BGTZ;

        do_branch:

            gen_compute_branch(ctx, mips32_op, 4, rs, -1, imm << 1);

            *is_branch = 1;

            break;



            /* Traps */

        case TLTI:

            mips32_op = OPC_TLTI;

            goto do_trapi;

        case TGEI:

            mips32_op = OPC_TGEI;

            goto do_trapi;

        case TLTIU:

            mips32_op = OPC_TLTIU;

            goto do_trapi;

        case TGEIU:

            mips32_op = OPC_TGEIU;

            goto do_trapi;

        case TNEI:

            mips32_op = OPC_TNEI;

            goto do_trapi;

        case TEQI:

            mips32_op = OPC_TEQI;

        do_trapi:

            gen_trap(ctx, mips32_op, rs, -1, imm);

            break;



        case BNEZC:

        case BEQZC:

            gen_compute_branch(ctx, minor == BNEZC ? OPC_BNE : OPC_BEQ,

                               4, rs, 0, imm << 1);

            /* Compact branches don't have a delay slot, so just let

               the normal delay slot handling take us to the branch

               target. */

            break;

        case LUI:

            gen_logic_imm(env, OPC_LUI, rs, -1, imm);

            break;

        case SYNCI:

            break;

        case BC2F:

        case BC2T:

            /* COP2: Not implemented. */

            generate_exception_err(ctx, EXCP_CpU, 2);

            break;

        case BC1F:

            mips32_op = (ctx->opcode & (1 << 16)) ? OPC_BC1FANY2 : OPC_BC1F;

            goto do_cp1branch;

        case BC1T:

            mips32_op = (ctx->opcode & (1 << 16)) ? OPC_BC1TANY2 : OPC_BC1T;

            goto do_cp1branch;

        case BC1ANY4F:

            mips32_op = OPC_BC1FANY4;

            goto do_cp1mips3d;

        case BC1ANY4T:

            mips32_op = OPC_BC1TANY4;

        do_cp1mips3d:

            check_cop1x(ctx);

            check_insn(env, ctx, ASE_MIPS3D);

            /* Fall through */

        do_cp1branch:

            gen_compute_branch1(env, ctx, mips32_op,

                                (ctx->opcode >> 18) & 0x7, imm << 1);

            *is_branch = 1;

            break;

        case BPOSGE64:

        case BPOSGE32:

            /* MIPS DSP: not implemented */

            /* Fall through */

        default:

            MIPS_INVAL("pool32i");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case POOL32C:

        minor = (ctx->opcode >> 12) & 0xf;

        switch (minor) {

        case LWL:

            mips32_op = OPC_LWL;

            goto do_ld_lr;

        case SWL:

            mips32_op = OPC_SWL;

            goto do_st_lr;

        case LWR:

            mips32_op = OPC_LWR;

            goto do_ld_lr;

        case SWR:

            mips32_op = OPC_SWR;

            goto do_st_lr;

#if defined(TARGET_MIPS64)

        case LDL:

            mips32_op = OPC_LDL;

            goto do_ld_lr;

        case SDL:

            mips32_op = OPC_SDL;

            goto do_st_lr;

        case LDR:

            mips32_op = OPC_LDR;

            goto do_ld_lr;

        case SDR:

            mips32_op = OPC_SDR;

            goto do_st_lr;

        case LWU:

            mips32_op = OPC_LWU;

            goto do_ld_lr;

        case LLD:

            mips32_op = OPC_LLD;

            goto do_ld_lr;

#endif

        case LL:

            mips32_op = OPC_LL;

            goto do_ld_lr;

        do_ld_lr:

            gen_ld(env, ctx, mips32_op, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

        do_st_lr:

            gen_st(ctx, mips32_op, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

        case SC:

            gen_st_cond(ctx, OPC_SC, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

#if defined(TARGET_MIPS64)

        case SCD:

            gen_st_cond(ctx, OPC_SCD, rt, rs, SIMM(ctx->opcode, 0, 12));

            break;

#endif

        case PREF:

            /* Treat as no-op */

            break;

        default:

            MIPS_INVAL("pool32c");

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case ADDI32:

        mips32_op = OPC_ADDI;

        goto do_addi;

    case ADDIU32:

        mips32_op = OPC_ADDIU;

    do_addi:

        gen_arith_imm(env, ctx, mips32_op, rt, rs, imm);

        break;



        /* Logical operations */

    case ORI32:

        mips32_op = OPC_ORI;

        goto do_logici;

    case XORI32:

        mips32_op = OPC_XORI;

        goto do_logici;

    case ANDI32:

        mips32_op = OPC_ANDI;

    do_logici:

        gen_logic_imm(env, mips32_op, rt, rs, imm);

        break;



        /* Set less than immediate */

    case SLTI32:

        mips32_op = OPC_SLTI;

        goto do_slti;

    case SLTIU32:

        mips32_op = OPC_SLTIU;

    do_slti:

        gen_slt_imm(env, mips32_op, rt, rs, imm);

        break;

    case JALX32:

        offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

        gen_compute_branch(ctx, OPC_JALX, 4, rt, rs, offset);

        *is_branch = 1;

        break;

    case JALS32:

        offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 1;

        gen_compute_branch(ctx, OPC_JALS, 4, rt, rs, offset);

        *is_branch = 1;

        break;

    case BEQ32:

        gen_compute_branch(ctx, OPC_BEQ, 4, rt, rs, imm << 1);

        *is_branch = 1;

        break;

    case BNE32:

        gen_compute_branch(ctx, OPC_BNE, 4, rt, rs, imm << 1);

        *is_branch = 1;

        break;

    case J32:

        gen_compute_branch(ctx, OPC_J, 4, rt, rs,

                           (int32_t)(ctx->opcode & 0x3FFFFFF) << 1);

        *is_branch = 1;

        break;

    case JAL32:

        gen_compute_branch(ctx, OPC_JAL, 4, rt, rs,

                           (int32_t)(ctx->opcode & 0x3FFFFFF) << 1);

        *is_branch = 1;

        break;

        /* Floating point (COP1) */

    case LWC132:

        mips32_op = OPC_LWC1;

        goto do_cop1;

    case LDC132:

        mips32_op = OPC_LDC1;

        goto do_cop1;

    case SWC132:

        mips32_op = OPC_SWC1;

        goto do_cop1;

    case SDC132:

        mips32_op = OPC_SDC1;

    do_cop1:

        gen_cop1_ldst(env, ctx, mips32_op, rt, rs, imm);

        break;

    case ADDIUPC:

        {

            int reg = mmreg(ZIMM(ctx->opcode, 23, 3));

            int offset = SIMM(ctx->opcode, 0, 23) << 2;



            gen_addiupc(ctx, reg, offset, 0, 0);

        }

        break;

        /* Loads and stores */

    case LB32:

        mips32_op = OPC_LB;

        goto do_ld;

    case LBU32:

        mips32_op = OPC_LBU;

        goto do_ld;

    case LH32:

        mips32_op = OPC_LH;

        goto do_ld;

    case LHU32:

        mips32_op = OPC_LHU;

        goto do_ld;

    case LW32:

        mips32_op = OPC_LW;

        goto do_ld;

#ifdef TARGET_MIPS64

    case LD32:

        mips32_op = OPC_LD;

        goto do_ld;

    case SD32:

        mips32_op = OPC_SD;

        goto do_st;

#endif

    case SB32:

        mips32_op = OPC_SB;

        goto do_st;

    case SH32:

        mips32_op = OPC_SH;

        goto do_st;

    case SW32:

        mips32_op = OPC_SW;

        goto do_st;

    do_ld:

        gen_ld(env, ctx, mips32_op, rt, rs, imm);

        break;

    do_st:

        gen_st(ctx, mips32_op, rt, rs, imm);

        break;

    default:

        generate_exception(ctx, EXCP_RI);

        break;

    }

}