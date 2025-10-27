static void decode_rlc_opc(CPUTriCoreState *env, DisasContext *ctx,

                           uint32_t op1)

{

    int32_t const16;

    int r1, r2;



    const16 = MASK_OP_RLC_CONST16_SEXT(ctx->opcode);

    r1      = MASK_OP_RLC_S1(ctx->opcode);

    r2      = MASK_OP_RLC_D(ctx->opcode);



    switch (op1) {

    case OPC1_32_RLC_ADDI:

        gen_addi_d(cpu_gpr_d[r2], cpu_gpr_d[r1], const16);

        break;

    case OPC1_32_RLC_ADDIH:

        gen_addi_d(cpu_gpr_d[r2], cpu_gpr_d[r1], const16 << 16);

        break;

    case OPC1_32_RLC_ADDIH_A:

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r1], const16 << 16);

        break;

    case OPC1_32_RLC_MFCR:

        const16 = MASK_OP_RLC_CONST16(ctx->opcode);

        gen_mfcr(env, cpu_gpr_d[r2], const16);

        break;

    case OPC1_32_RLC_MOV:

        tcg_gen_movi_tl(cpu_gpr_d[r2], const16);

        break;

    case OPC1_32_RLC_MOV_64:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            if ((r2 & 0x1) != 0) {

                /* TODO: raise OPD trap */

            }

            tcg_gen_movi_tl(cpu_gpr_d[r2], const16);

            tcg_gen_movi_tl(cpu_gpr_d[r2+1], const16 >> 15);

        } else {

            /* TODO: raise illegal opcode trap */

        }

        break;

    case OPC1_32_RLC_MOV_U:

        const16 = MASK_OP_RLC_CONST16(ctx->opcode);

        tcg_gen_movi_tl(cpu_gpr_d[r2], const16);

        break;

    case OPC1_32_RLC_MOV_H:

        tcg_gen_movi_tl(cpu_gpr_d[r2], const16 << 16);

        break;

    case OPC1_32_RLC_MOVH_A:

        tcg_gen_movi_tl(cpu_gpr_a[r2], const16 << 16);

        break;

    case OPC1_32_RLC_MTCR:

        const16 = MASK_OP_RLC_CONST16(ctx->opcode);

        gen_mtcr(env, ctx, cpu_gpr_d[r1], const16);

        break;

    }

}
