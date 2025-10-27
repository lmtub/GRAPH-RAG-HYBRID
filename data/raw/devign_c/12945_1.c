static void decode_src_opc(CPUTriCoreState *env, DisasContext *ctx, int op1)

{

    int r1;

    int32_t const4;

    TCGv temp, temp2;



    r1 = MASK_OP_SRC_S1D(ctx->opcode);

    const4 = MASK_OP_SRC_CONST4_SEXT(ctx->opcode);



    switch (op1) {

    case OPC1_16_SRC_ADD:

        gen_addi_d(cpu_gpr_d[r1], cpu_gpr_d[r1], const4);

        break;

    case OPC1_16_SRC_ADD_A15:

        gen_addi_d(cpu_gpr_d[r1], cpu_gpr_d[15], const4);

        break;

    case OPC1_16_SRC_ADD_15A:

        gen_addi_d(cpu_gpr_d[15], cpu_gpr_d[r1], const4);

        break;

    case OPC1_16_SRC_ADD_A:

        tcg_gen_addi_tl(cpu_gpr_a[r1], cpu_gpr_a[r1], const4);

        break;

    case OPC1_16_SRC_CADD:

        gen_condi_add(TCG_COND_NE, cpu_gpr_d[r1], const4, cpu_gpr_d[r1],

                      cpu_gpr_d[15]);

        break;

    case OPC1_16_SRC_CADDN:

        gen_condi_add(TCG_COND_EQ, cpu_gpr_d[r1], const4, cpu_gpr_d[r1],

                      cpu_gpr_d[15]);

        break;

    case OPC1_16_SRC_CMOV:

        temp = tcg_const_tl(0);

        temp2 = tcg_const_tl(const4);

        tcg_gen_movcond_tl(TCG_COND_NE, cpu_gpr_d[r1], cpu_gpr_d[15], temp,

                           temp2, cpu_gpr_d[r1]);

        tcg_temp_free(temp);

        tcg_temp_free(temp2);

        break;

    case OPC1_16_SRC_CMOVN:

        temp = tcg_const_tl(0);

        temp2 = tcg_const_tl(const4);

        tcg_gen_movcond_tl(TCG_COND_EQ, cpu_gpr_d[r1], cpu_gpr_d[15], temp,

                           temp2, cpu_gpr_d[r1]);

        tcg_temp_free(temp);

        tcg_temp_free(temp2);

        break;

    case OPC1_16_SRC_EQ:

        tcg_gen_setcondi_tl(TCG_COND_EQ, cpu_gpr_d[15], cpu_gpr_d[r1],

                            const4);

        break;

    case OPC1_16_SRC_LT:

        tcg_gen_setcondi_tl(TCG_COND_LT, cpu_gpr_d[15], cpu_gpr_d[r1],

                            const4);

        break;

    case OPC1_16_SRC_MOV:

        tcg_gen_movi_tl(cpu_gpr_d[r1], const4);

        break;

    case OPC1_16_SRC_MOV_A:

        const4 = MASK_OP_SRC_CONST4(ctx->opcode);

        tcg_gen_movi_tl(cpu_gpr_a[r1], const4);

        break;

    case OPC1_16_SRC_MOV_E:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            tcg_gen_movi_tl(cpu_gpr_d[r1], const4);

            tcg_gen_sari_tl(cpu_gpr_d[r1+1], cpu_gpr_d[r1], 31);

        } /* TODO: else raise illegal opcode trap */

        break;

    case OPC1_16_SRC_SH:

        gen_shi(cpu_gpr_d[r1], cpu_gpr_d[r1], const4);

        break;

    case OPC1_16_SRC_SHA:

        gen_shaci(cpu_gpr_d[r1], cpu_gpr_d[r1], const4);

        break;

    }

}
