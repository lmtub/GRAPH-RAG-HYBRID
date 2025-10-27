static void gen_msa_branch(CPUMIPSState *env, DisasContext *ctx, uint32_t op1)

{

    uint8_t df = (ctx->opcode >> 21) & 0x3;

    uint8_t wt = (ctx->opcode >> 16) & 0x1f;

    int64_t s16 = (int16_t)ctx->opcode;



    check_msa_access(ctx);



    if (ctx->insn_flags & ISA_MIPS32R6 && ctx->hflags & MIPS_HFLAG_BMASK) {

        generate_exception_end(ctx, EXCP_RI);

        return;

    }

    switch (op1) {

    case OPC_BZ_V:

    case OPC_BNZ_V:

        {

            TCGv_i64 t0 = tcg_temp_new_i64();

            tcg_gen_or_i64(t0, msa_wr_d[wt<<1], msa_wr_d[(wt<<1)+1]);

            tcg_gen_setcondi_i64((op1 == OPC_BZ_V) ?

                    TCG_COND_EQ : TCG_COND_NE, t0, t0, 0);

            tcg_gen_trunc_i64_tl(bcond, t0);

            tcg_temp_free_i64(t0);

        }

        break;

    case OPC_BZ_B:

    case OPC_BZ_H:

    case OPC_BZ_W:

    case OPC_BZ_D:

        gen_check_zero_element(bcond, df, wt);

        break;

    case OPC_BNZ_B:

    case OPC_BNZ_H:

    case OPC_BNZ_W:

    case OPC_BNZ_D:

        gen_check_zero_element(bcond, df, wt);

        tcg_gen_setcondi_tl(TCG_COND_EQ, bcond, bcond, 0);

        break;

    }



    ctx->btarget = ctx->pc + (s16 << 2) + 4;



    ctx->hflags |= MIPS_HFLAG_BC;

    ctx->hflags |= MIPS_HFLAG_BDS32;

}
