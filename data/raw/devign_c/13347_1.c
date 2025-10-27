static void decode_abs_store(CPUTriCoreState *env, DisasContext *ctx)

{

    int32_t op2;

    int32_t r1;

    uint32_t address;

    TCGv temp;



    r1 = MASK_OP_ABS_S1D(ctx->opcode);

    address = MASK_OP_ABS_OFF18(ctx->opcode);

    op2 = MASK_OP_ABS_OP2(ctx->opcode);



    temp = tcg_const_i32(EA_ABS_FORMAT(address));



    switch (op2) {

    case OPC2_32_ABS_ST_A:

        tcg_gen_qemu_st_tl(cpu_gpr_a[r1], temp, ctx->mem_idx, MO_LESL);

        break;

    case OPC2_32_ABS_ST_D:

        gen_st_2regs_64(cpu_gpr_d[r1+1], cpu_gpr_d[r1], temp, ctx);

        break;

    case OPC2_32_ABS_ST_DA:

        gen_st_2regs_64(cpu_gpr_a[r1+1], cpu_gpr_a[r1], temp, ctx);

        break;

    case OPC2_32_ABS_ST_W:

        tcg_gen_qemu_st_tl(cpu_gpr_d[r1], temp, ctx->mem_idx, MO_LESL);

        break;



    }

    tcg_temp_free(temp);

}
