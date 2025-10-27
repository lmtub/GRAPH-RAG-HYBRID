static int decode_micromips_opc (CPUMIPSState *env, DisasContext *ctx, int *is_branch)

{

    uint32_t op;



    /* make sure instructions are on a halfword boundary */

    if (ctx->pc & 0x1) {

        env->CP0_BadVAddr = ctx->pc;

        generate_exception(ctx, EXCP_AdEL);

        ctx->bstate = BS_STOP;

        return 2;

    }



    op = (ctx->opcode >> 10) & 0x3f;

    /* Enforce properly-sized instructions in a delay slot */

    if (ctx->hflags & MIPS_HFLAG_BMASK) {

        int bits = ctx->hflags & MIPS_HFLAG_BMASK_EXT;



        switch (op) {

        case POOL32A:

        case POOL32B:

        case POOL32I:

        case POOL32C:

        case ADDI32:

        case ADDIU32:

        case ORI32:

        case XORI32:

        case SLTI32:

        case SLTIU32:

        case ANDI32:

        case JALX32:

        case LBU32:

        case LHU32:

        case POOL32F:

        case JALS32:

        case BEQ32:

        case BNE32:

        case J32:

        case JAL32:

        case SB32:

        case SH32:

        case POOL32S:

        case ADDIUPC:

        case SWC132:

        case SDC132:

        case SD32:

        case SW32:

        case LB32:

        case LH32:

        case DADDIU32:

        case LWC132:

        case LDC132:

        case LD32:

        case LW32:

            if (bits & MIPS_HFLAG_BDS16) {

                generate_exception(ctx, EXCP_RI);

                /* Just stop translation; the user is confused.  */

                ctx->bstate = BS_STOP;

                return 2;

            }

            break;

        case POOL16A:

        case POOL16B:

        case POOL16C:

        case LWGP16:

        case POOL16F:

        case LBU16:

        case LHU16:

        case LWSP16:

        case LW16:

        case SB16:

        case SH16:

        case SWSP16:

        case SW16:

        case MOVE16:

        case ANDI16:

        case POOL16D:

        case POOL16E:

        case BEQZ16:

        case BNEZ16:

        case B16:

        case LI16:

            if (bits & MIPS_HFLAG_BDS32) {

                generate_exception(ctx, EXCP_RI);

                /* Just stop translation; the user is confused.  */

                ctx->bstate = BS_STOP;

                return 2;

            }

            break;

        default:

            break;

        }

    }

    switch (op) {

    case POOL16A:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rs1 = mmreg(uMIPS_RS1(ctx->opcode));

            int rs2 = mmreg(uMIPS_RS2(ctx->opcode));

            uint32_t opc = 0;



            switch (ctx->opcode & 0x1) {

            case ADDU16:

                opc = OPC_ADDU;

                break;

            case SUBU16:

                opc = OPC_SUBU;

                break;

            }



            gen_arith(ctx, opc, rd, rs1, rs2);

        }

        break;

    case POOL16B:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rs = mmreg(uMIPS_RS(ctx->opcode));

            int amount = (ctx->opcode >> 1) & 0x7;

            uint32_t opc = 0;

            amount = amount == 0 ? 8 : amount;



            switch (ctx->opcode & 0x1) {

            case SLL16:

                opc = OPC_SLL;

                break;

            case SRL16:

                opc = OPC_SRL;

                break;

            }



            gen_shift_imm(ctx, opc, rd, rs, amount);

        }

        break;

    case POOL16C:

        gen_pool16c_insn(ctx, is_branch);

        break;

    case LWGP16:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rb = 28;            /* GP */

            int16_t offset = SIMM(ctx->opcode, 0, 7) << 2;



            gen_ld(ctx, OPC_LW, rd, rb, offset);

        }

        break;

    case POOL16F:

        if (ctx->opcode & 1) {

            generate_exception(ctx, EXCP_RI);

        } else {

            /* MOVEP */

            int enc_dest = uMIPS_RD(ctx->opcode);

            int enc_rt = uMIPS_RS2(ctx->opcode);

            int enc_rs = uMIPS_RS1(ctx->opcode);

            int rd, rs, re, rt;

            static const int rd_enc[] = { 5, 5, 6, 4, 4, 4, 4, 4 };

            static const int re_enc[] = { 6, 7, 7, 21, 22, 5, 6, 7 };

            static const int rs_rt_enc[] = { 0, 17, 2, 3, 16, 18, 19, 20 };



            rd = rd_enc[enc_dest];

            re = re_enc[enc_dest];

            rs = rs_rt_enc[enc_rs];

            rt = rs_rt_enc[enc_rt];



            gen_arith_imm(ctx, OPC_ADDIU, rd, rs, 0);

            gen_arith_imm(ctx, OPC_ADDIU, re, rt, 0);

        }

        break;

    case LBU16:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4);

            offset = (offset == 0xf ? -1 : offset);



            gen_ld(ctx, OPC_LBU, rd, rb, offset);

        }

        break;

    case LHU16:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4) << 1;



            gen_ld(ctx, OPC_LHU, rd, rb, offset);

        }

        break;

    case LWSP16:

        {

            int rd = (ctx->opcode >> 5) & 0x1f;

            int rb = 29;            /* SP */

            int16_t offset = ZIMM(ctx->opcode, 0, 5) << 2;



            gen_ld(ctx, OPC_LW, rd, rb, offset);

        }

        break;

    case LW16:

        {

            int rd = mmreg(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4) << 2;



            gen_ld(ctx, OPC_LW, rd, rb, offset);

        }

        break;

    case SB16:

        {

            int rd = mmreg2(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4);



            gen_st(ctx, OPC_SB, rd, rb, offset);

        }

        break;

    case SH16:

        {

            int rd = mmreg2(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4) << 1;



            gen_st(ctx, OPC_SH, rd, rb, offset);

        }

        break;

    case SWSP16:

        {

            int rd = (ctx->opcode >> 5) & 0x1f;

            int rb = 29;            /* SP */

            int16_t offset = ZIMM(ctx->opcode, 0, 5) << 2;



            gen_st(ctx, OPC_SW, rd, rb, offset);

        }

        break;

    case SW16:

        {

            int rd = mmreg2(uMIPS_RD(ctx->opcode));

            int rb = mmreg(uMIPS_RS(ctx->opcode));

            int16_t offset = ZIMM(ctx->opcode, 0, 4) << 2;



            gen_st(ctx, OPC_SW, rd, rb, offset);

        }

        break;

    case MOVE16:

        {

            int rd = uMIPS_RD5(ctx->opcode);

            int rs = uMIPS_RS5(ctx->opcode);



            gen_arith_imm(ctx, OPC_ADDIU, rd, rs, 0);

        }

        break;

    case ANDI16:

        gen_andi16(ctx);

        break;

    case POOL16D:

        switch (ctx->opcode & 0x1) {

        case ADDIUS5:

            gen_addius5(ctx);

            break;

        case ADDIUSP:

            gen_addiusp(ctx);

            break;

        }

        break;

    case POOL16E:

        switch (ctx->opcode & 0x1) {

        case ADDIUR2:

            gen_addiur2(ctx);

            break;

        case ADDIUR1SP:

            gen_addiur1sp(ctx);

            break;

        }

        break;

    case B16:

        gen_compute_branch(ctx, OPC_BEQ, 2, 0, 0,

                           SIMM(ctx->opcode, 0, 10) << 1);

        *is_branch = 1;

        break;

    case BNEZ16:

    case BEQZ16:

        gen_compute_branch(ctx, op == BNEZ16 ? OPC_BNE : OPC_BEQ, 2,

                           mmreg(uMIPS_RD(ctx->opcode)),

                           0, SIMM(ctx->opcode, 0, 7) << 1);

        *is_branch = 1;

        break;

    case LI16:

        {

            int reg = mmreg(uMIPS_RD(ctx->opcode));

            int imm = ZIMM(ctx->opcode, 0, 7);



            imm = (imm == 0x7f ? -1 : imm);

            tcg_gen_movi_tl(cpu_gpr[reg], imm);

        }

        break;

    case RES_20:

    case RES_28:

    case RES_29:

    case RES_30:

    case RES_31:

    case RES_38:

    case RES_39:

        generate_exception(ctx, EXCP_RI);

        break;

    default:

        decode_micromips32_opc (env, ctx, op, is_branch);

        return 4;

    }



    return 2;

}
