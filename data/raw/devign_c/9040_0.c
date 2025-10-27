static void decode_opc_special3(CPUMIPSState *env, DisasContext *ctx)

{

    int rs, rt, rd, sa;

    uint32_t op1, op2;



    rs = (ctx->opcode >> 21) & 0x1f;

    rt = (ctx->opcode >> 16) & 0x1f;

    rd = (ctx->opcode >> 11) & 0x1f;

    sa = (ctx->opcode >> 6) & 0x1f;



    op1 = MASK_SPECIAL3(ctx->opcode);

    switch (op1) {

    case OPC_EXT:

    case OPC_INS:

        check_insn(ctx, ISA_MIPS32R2);

        gen_bitops(ctx, op1, rt, rs, sa, rd);

        break;

    case OPC_BSHFL:

        op2 = MASK_BSHFL(ctx->opcode);

        switch (op2) {

        case OPC_ALIGN ... OPC_ALIGN_END:

        case OPC_BITSWAP:

            check_insn(ctx, ISA_MIPS32R6);

            decode_opc_special3_r6(env, ctx);

            break;

        default:

            check_insn(ctx, ISA_MIPS32R2);

            gen_bshfl(ctx, op2, rt, rd);

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

        op2 = MASK_DBSHFL(ctx->opcode);

        switch (op2) {

        case OPC_DALIGN ... OPC_DALIGN_END:

        case OPC_DBITSWAP:

            check_insn(ctx, ISA_MIPS32R6);

            decode_opc_special3_r6(env, ctx);

            break;

        default:

            check_insn(ctx, ISA_MIPS64R2);

            check_mips_64(ctx);

            op2 = MASK_DBSHFL(ctx->opcode);

            gen_bshfl(ctx, op2, rt, rd);

            break;

        }

        break;

#endif

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



            gen_load_gpr(t0, rs);

            gen_helper_yield(t0, cpu_env, t0);

            gen_store_gpr(t0, rd);

            tcg_temp_free(t0);

        }

        break;

    default:

        if (ctx->insn_flags & ISA_MIPS32R6) {

            decode_opc_special3_r6(env, ctx);

        } else {

            decode_opc_special3_legacy(env, ctx);

        }

    }

}
