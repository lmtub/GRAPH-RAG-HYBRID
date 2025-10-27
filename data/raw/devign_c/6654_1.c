static void decode_bol_opc(CPUTriCoreState *env, DisasContext *ctx, int32_t op1)

{

    int r1, r2;

    int32_t address;

    TCGv temp;



    r1 = MASK_OP_BOL_S1D(ctx->opcode);

    r2 = MASK_OP_BOL_S2(ctx->opcode);

    address = MASK_OP_BOL_OFF16_SEXT(ctx->opcode);



    switch (op1) {

    case OPC1_32_BOL_LD_A_LONGOFF:

        temp = tcg_temp_new();

        tcg_gen_addi_tl(temp, cpu_gpr_a[r2], address);

        tcg_gen_qemu_ld_tl(cpu_gpr_a[r1], temp, ctx->mem_idx, MO_LEUL);

        tcg_temp_free(temp);

        break;

    case OPC1_32_BOL_LD_W_LONGOFF:

        temp = tcg_temp_new();

        tcg_gen_addi_tl(temp, cpu_gpr_a[r2], address);

        tcg_gen_qemu_ld_tl(cpu_gpr_d[r1], temp, ctx->mem_idx, MO_LEUL);

        tcg_temp_free(temp);

        break;

    case OPC1_32_BOL_LEA_LONGOFF:

        tcg_gen_addi_tl(cpu_gpr_a[r1], cpu_gpr_a[r2], address);

        break;

    case OPC1_32_BOL_ST_A_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_st(ctx, cpu_gpr_a[r1], cpu_gpr_a[r2], address, MO_LEUL);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_ST_W_LONGOFF:

        gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_LEUL);

        break;

    case OPC1_32_BOL_LD_B_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_ld(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_SB);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_LD_BU_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_ld(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_UB);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_LD_H_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_ld(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_LESW);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_LD_HU_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_ld(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_LEUW);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_ST_B_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_SB);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    case OPC1_32_BOL_ST_H_LONGOFF:

        if (tricore_feature(env, TRICORE_FEATURE_16)) {

            gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], address, MO_LESW);

        } else {

            /* raise illegal opcode trap */

        }

        break;

    }

}
