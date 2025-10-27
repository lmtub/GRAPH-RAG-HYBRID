static int decode_micromips_opc (CPUMIPSState *env, DisasContext *ctx)

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

    if (ctx->hflags & MIPS_HFLAG_BDS_STRICT) {

        switch (op & 0x7) { /* MSB-3..MSB-5 */

        case 0:

        /* POOL32A, POOL32B, POOL32I, POOL32C */

        case 4:

        /* ADDI32, ADDIU32, ORI32, XORI32, SLTI32, SLTIU32, ANDI32, JALX32 */

        case 5:

        /* LBU32, LHU32, POOL32F, JALS32, BEQ32, BNE32, J32, JAL32 */

        case 6:

        /* SB32, SH32, ADDIUPC, SWC132, SDC132, SW32 */

        case 7:

        /* LB32, LH32, LWC132, LDC132, LW32 */

            if (ctx->hflags & MIPS_HFLAG_BDS16) {

                generate_exception(ctx, EXCP_RI);

                /* Just stop translation; the user is confused.  */

                ctx->bstate = BS_STOP;

                return 2;

            }

            break;

        case 1:

        /* POOL16A, POOL16B, POOL16C, LWGP16, POOL16F */

        case 2:

        /* LBU16, LHU16, LWSP16, LW16, SB16, SH16, SWSP16, SW16 */

        case 3:

        /* MOVE16, ANDI16, POOL16D, POOL16E, BEQZ16, BNEZ16, B16, LI16 */

            if (ctx->hflags & MIPS_HFLAG_BDS32) {

                generate_exception(ctx, EXCP_RI);

                /* Just stop translation; the user is confused.  */

                ctx->bstate = BS_STOP;

                return 2;

            }

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

        gen_pool16c_insn(ctx);

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

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

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



            gen_arith(ctx, OPC_ADDU, rd, rs, 0);

            gen_arith(ctx, OPC_ADDU, re, rt, 0);

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



            gen_arith(ctx, OPC_ADDU, rd, rs, 0);

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

                           SIMM(ctx->opcode, 0, 10) << 1, 4);

        break;

    case BNEZ16:

    case BEQZ16:

        gen_compute_branch(ctx, op == BNEZ16 ? OPC_BNE : OPC_BEQ, 2,

                           mmreg(uMIPS_RD(ctx->opcode)),

                           0, SIMM(ctx->opcode, 0, 7) << 1, 4);

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

        decode_micromips32_opc(env, ctx);

        return 4;

    }



    return 2;

}
