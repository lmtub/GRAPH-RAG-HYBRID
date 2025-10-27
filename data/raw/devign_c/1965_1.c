static void decode_bo_addrmode_post_pre_base(CPUTriCoreState *env,

                                             DisasContext *ctx)

{

    uint32_t op2;

    uint32_t off10;

    int32_t r1, r2;

    TCGv temp;



    r1 = MASK_OP_BO_S1D(ctx->opcode);

    r2  = MASK_OP_BO_S2(ctx->opcode);

    off10 = MASK_OP_BO_OFF10_SEXT(ctx->opcode);

    op2 = MASK_OP_BO_OP2(ctx->opcode);



    switch (op2) {

    case OPC2_32_BO_CACHEA_WI_SHORTOFF:

    case OPC2_32_BO_CACHEA_W_SHORTOFF:

    case OPC2_32_BO_CACHEA_I_SHORTOFF:

        /* instruction to access the cache */

        break;

    case OPC2_32_BO_CACHEA_WI_POSTINC:

    case OPC2_32_BO_CACHEA_W_POSTINC:

    case OPC2_32_BO_CACHEA_I_POSTINC:

        /* instruction to access the cache, but we still need to handle

           the addressing mode */

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_CACHEA_WI_PREINC:

    case OPC2_32_BO_CACHEA_W_PREINC:

    case OPC2_32_BO_CACHEA_I_PREINC:

        /* instruction to access the cache, but we still need to handle

           the addressing mode */

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_CACHEI_WI_SHORTOFF:

    case OPC2_32_BO_CACHEI_W_SHORTOFF:

        /* TODO: Raise illegal opcode trap,

                 if !tricore_feature(TRICORE_FEATURE_131) */

        break;

    case OPC2_32_BO_CACHEI_W_POSTINC:

    case OPC2_32_BO_CACHEI_WI_POSTINC:

        if (tricore_feature(env, TRICORE_FEATURE_131)) {

            tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        } /* TODO: else raise illegal opcode trap */

        break;

    case OPC2_32_BO_CACHEI_W_PREINC:

    case OPC2_32_BO_CACHEI_WI_PREINC:

        if (tricore_feature(env, TRICORE_FEATURE_131)) {

            tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        } /* TODO: else raise illegal opcode trap */

        break;

    case OPC2_32_BO_ST_A_SHORTOFF:

        gen_offset_st(ctx, cpu_gpr_a[r1], cpu_gpr_a[r2], off10, MO_LESL);

        break;

    case OPC2_32_BO_ST_A_POSTINC:

        tcg_gen_qemu_st_tl(cpu_gpr_a[r1], cpu_gpr_a[r2], ctx->mem_idx,

                           MO_LESL);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_A_PREINC:

        gen_st_preincr(ctx, cpu_gpr_a[r1], cpu_gpr_a[r2], off10, MO_LESL);

        break;

    case OPC2_32_BO_ST_B_SHORTOFF:

        gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_UB);

        break;

    case OPC2_32_BO_ST_B_POSTINC:

        tcg_gen_qemu_st_tl(cpu_gpr_d[r1], cpu_gpr_a[r2], ctx->mem_idx,

                           MO_UB);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_B_PREINC:

        gen_st_preincr(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_UB);

        break;

    case OPC2_32_BO_ST_D_SHORTOFF:

        gen_offset_st_2regs(cpu_gpr_d[r1+1], cpu_gpr_d[r1], cpu_gpr_a[r2],

                            off10, ctx);

        break;

    case OPC2_32_BO_ST_D_POSTINC:

        gen_st_2regs_64(cpu_gpr_d[r1+1], cpu_gpr_d[r1], cpu_gpr_a[r2], ctx);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_D_PREINC:

        temp = tcg_temp_new();

        tcg_gen_addi_tl(temp, cpu_gpr_a[r2], off10);

        gen_st_2regs_64(cpu_gpr_d[r1+1], cpu_gpr_d[r1], temp, ctx);

        tcg_gen_mov_tl(cpu_gpr_a[r2], temp);

        tcg_temp_free(temp);

        break;

    case OPC2_32_BO_ST_DA_SHORTOFF:

        gen_offset_st_2regs(cpu_gpr_a[r1+1], cpu_gpr_a[r1], cpu_gpr_a[r2],

                            off10, ctx);

        break;

    case OPC2_32_BO_ST_DA_POSTINC:

        gen_st_2regs_64(cpu_gpr_a[r1+1], cpu_gpr_a[r1], cpu_gpr_a[r2], ctx);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_DA_PREINC:

        temp = tcg_temp_new();

        tcg_gen_addi_tl(temp, cpu_gpr_a[r2], off10);

        gen_st_2regs_64(cpu_gpr_a[r1+1], cpu_gpr_a[r1], temp, ctx);

        tcg_gen_mov_tl(cpu_gpr_a[r2], temp);

        tcg_temp_free(temp);

        break;

    case OPC2_32_BO_ST_H_SHORTOFF:

        gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_LEUW);

        break;

    case OPC2_32_BO_ST_H_POSTINC:

        tcg_gen_qemu_st_tl(cpu_gpr_d[r1], cpu_gpr_a[r2], ctx->mem_idx,

                           MO_LEUW);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_H_PREINC:

        gen_st_preincr(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_LEUW);

        break;

    case OPC2_32_BO_ST_Q_SHORTOFF:

        temp = tcg_temp_new();

        tcg_gen_shri_tl(temp, cpu_gpr_d[r1], 16);

        gen_offset_st(ctx, temp, cpu_gpr_a[r2], off10, MO_LEUW);

        tcg_temp_free(temp);

        break;

    case OPC2_32_BO_ST_Q_POSTINC:

        temp = tcg_temp_new();

        tcg_gen_shri_tl(temp, cpu_gpr_d[r1], 16);

        tcg_gen_qemu_st_tl(temp, cpu_gpr_a[r2], ctx->mem_idx,

                           MO_LEUW);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        tcg_temp_free(temp);

        break;

    case OPC2_32_BO_ST_Q_PREINC:

        temp = tcg_temp_new();

        tcg_gen_shri_tl(temp, cpu_gpr_d[r1], 16);

        gen_st_preincr(ctx, temp, cpu_gpr_a[r2], off10, MO_LEUW);

        tcg_temp_free(temp);

        break;

    case OPC2_32_BO_ST_W_SHORTOFF:

        gen_offset_st(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_LEUL);

        break;

    case OPC2_32_BO_ST_W_POSTINC:

        tcg_gen_qemu_st_tl(cpu_gpr_d[r1], cpu_gpr_a[r2], ctx->mem_idx,

                           MO_LEUL);

        tcg_gen_addi_tl(cpu_gpr_a[r2], cpu_gpr_a[r2], off10);

        break;

    case OPC2_32_BO_ST_W_PREINC:

        gen_st_preincr(ctx, cpu_gpr_d[r1], cpu_gpr_a[r2], off10, MO_LEUL);

        break;

    }

}
