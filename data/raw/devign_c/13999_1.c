static void gen_msa(CPUMIPSState *env, DisasContext *ctx)

{

    uint32_t opcode = ctx->opcode;

    check_insn(ctx, ASE_MSA);

    check_msa_access(ctx);



    switch (MASK_MSA_MINOR(opcode)) {

    case OPC_MSA_I8_00:

    case OPC_MSA_I8_01:

    case OPC_MSA_I8_02:

        gen_msa_i8(env, ctx);

        break;

    case OPC_MSA_I5_06:

    case OPC_MSA_I5_07:

        gen_msa_i5(env, ctx);

        break;

    case OPC_MSA_BIT_09:

    case OPC_MSA_BIT_0A:

        gen_msa_bit(env, ctx);

        break;

    case OPC_MSA_3R_0D:

    case OPC_MSA_3R_0E:

    case OPC_MSA_3R_0F:

    case OPC_MSA_3R_10:

    case OPC_MSA_3R_11:

    case OPC_MSA_3R_12:

    case OPC_MSA_3R_13:

    case OPC_MSA_3R_14:

    case OPC_MSA_3R_15:

        gen_msa_3r(env, ctx);

        break;

    case OPC_MSA_ELM:

        gen_msa_elm(env, ctx);

        break;

    case OPC_MSA_3RF_1A:

    case OPC_MSA_3RF_1B:

    case OPC_MSA_3RF_1C:

        gen_msa_3rf(env, ctx);

        break;

    case OPC_MSA_VEC:

        gen_msa_vec(env, ctx);

        break;

    case OPC_LD_B:

    case OPC_LD_H:

    case OPC_LD_W:

    case OPC_LD_D:

    case OPC_ST_B:

    case OPC_ST_H:

    case OPC_ST_W:

    case OPC_ST_D:

        {

            int32_t s10 = sextract32(ctx->opcode, 16, 10);

            uint8_t rs = (ctx->opcode >> 11) & 0x1f;

            uint8_t wd = (ctx->opcode >> 6) & 0x1f;

            uint8_t df = (ctx->opcode >> 0) & 0x3;



            TCGv_i32 tdf = tcg_const_i32(df);

            TCGv_i32 twd = tcg_const_i32(wd);

            TCGv_i32 trs = tcg_const_i32(rs);

            TCGv_i32 ts10 = tcg_const_i32(s10);



            switch (MASK_MSA_MINOR(opcode)) {

            case OPC_LD_B:

            case OPC_LD_H:

            case OPC_LD_W:

            case OPC_LD_D:


                gen_helper_msa_ld_df(cpu_env, tdf, twd, trs, ts10);

                break;

            case OPC_ST_B:

            case OPC_ST_H:

            case OPC_ST_W:

            case OPC_ST_D:


                gen_helper_msa_st_df(cpu_env, tdf, twd, trs, ts10);

                break;

            }



            tcg_temp_free_i32(twd);

            tcg_temp_free_i32(tdf);

            tcg_temp_free_i32(trs);

            tcg_temp_free_i32(ts10);

        }

        break;

    default:

        MIPS_INVAL("MSA instruction");

        generate_exception(ctx, EXCP_RI);

        break;

    }



}