static void decode_32Bit_opc(CPUTriCoreState *env, DisasContext *ctx)

{

    int op1;

    int32_t r1, r2, r3;

    int32_t address, const16;

    int8_t b, const4;

    int32_t bpos;

    TCGv temp, temp2, temp3;



    op1 = MASK_OP_MAJOR(ctx->opcode);



    /* handle JNZ.T opcode only being 7 bit long */

    if (unlikely((op1 & 0x7f) == OPCM_32_BRN_JTT)) {

        op1 = OPCM_32_BRN_JTT;

    }



    switch (op1) {

/* ABS-format */

    case OPCM_32_ABS_LDW:

        decode_abs_ldw(env, ctx);


    case OPCM_32_ABS_LDB:

        decode_abs_ldb(env, ctx);


    case OPCM_32_ABS_LDMST_SWAP:

        decode_abs_ldst_swap(env, ctx);


    case OPCM_32_ABS_LDST_CONTEXT:

        decode_abs_ldst_context(env, ctx);


    case OPCM_32_ABS_STORE:

        decode_abs_store(env, ctx);


    case OPCM_32_ABS_STOREB_H:

        decode_abs_storeb_h(env, ctx);


    case OPC1_32_ABS_STOREQ:

        address = MASK_OP_ABS_OFF18(ctx->opcode);

        r1 = MASK_OP_ABS_S1D(ctx->opcode);

        temp = tcg_const_i32(EA_ABS_FORMAT(address));

        temp2 = tcg_temp_new();



        tcg_gen_shri_tl(temp2, cpu_gpr_d[r1], 16);

        tcg_gen_qemu_st_tl(temp2, temp, ctx->mem_idx, MO_LEUW);



        tcg_temp_free(temp2);

        tcg_temp_free(temp);


    case OPC1_32_ABS_LD_Q:

        address = MASK_OP_ABS_OFF18(ctx->opcode);

        r1 = MASK_OP_ABS_S1D(ctx->opcode);

        temp = tcg_const_i32(EA_ABS_FORMAT(address));



        tcg_gen_qemu_ld_tl(cpu_gpr_d[r1], temp, ctx->mem_idx, MO_LEUW);

        tcg_gen_shli_tl(cpu_gpr_d[r1], cpu_gpr_d[r1], 16);



        tcg_temp_free(temp);


    case OPC1_32_ABS_LEA:

        address = MASK_OP_ABS_OFF18(ctx->opcode);

        r1 = MASK_OP_ABS_S1D(ctx->opcode);

        tcg_gen_movi_tl(cpu_gpr_a[r1], EA_ABS_FORMAT(address));


/* ABSB-format */

    case OPC1_32_ABSB_ST_T:

        address = MASK_OP_ABS_OFF18(ctx->opcode);

        b = MASK_OP_ABSB_B(ctx->opcode);

        bpos = MASK_OP_ABSB_BPOS(ctx->opcode);



        temp = tcg_const_i32(EA_ABS_FORMAT(address));

        temp2 = tcg_temp_new();



        tcg_gen_qemu_ld_tl(temp2, temp, ctx->mem_idx, MO_UB);

        tcg_gen_andi_tl(temp2, temp2, ~(0x1u << bpos));

        tcg_gen_ori_tl(temp2, temp2, (b << bpos));

        tcg_gen_qemu_st_tl(temp2, temp, ctx->mem_idx, MO_UB);



        tcg_temp_free(temp);

        tcg_temp_free(temp2);


/* B-format */

    case OPC1_32_B_CALL:

    case OPC1_32_B_CALLA:

    case OPC1_32_B_J:

    case OPC1_32_B_JA:

    case OPC1_32_B_JL:

    case OPC1_32_B_JLA:

        address = MASK_OP_B_DISP24(ctx->opcode);

        gen_compute_branch(ctx, op1, 0, 0, 0, address);


/* Bit-format */

    case OPCM_32_BIT_ANDACC:

        decode_bit_andacc(env, ctx);


    case OPCM_32_BIT_LOGICAL_T1:

        decode_bit_logical_t(env, ctx);


    case OPCM_32_BIT_INSERT:

        decode_bit_insert(env, ctx);


    case OPCM_32_BIT_LOGICAL_T2:

        decode_bit_logical_t2(env, ctx);


    case OPCM_32_BIT_ORAND:

        decode_bit_orand(env, ctx);


    case OPCM_32_BIT_SH_LOGIC1:

        decode_bit_sh_logic1(env, ctx);


    case OPCM_32_BIT_SH_LOGIC2:

        decode_bit_sh_logic2(env, ctx);


    /* BO Format */

    case OPCM_32_BO_ADDRMODE_POST_PRE_BASE:

        decode_bo_addrmode_post_pre_base(env, ctx);


    case OPCM_32_BO_ADDRMODE_BITREVERSE_CIRCULAR:

        decode_bo_addrmode_bitreverse_circular(env, ctx);


    case OPCM_32_BO_ADDRMODE_LD_POST_PRE_BASE:

        decode_bo_addrmode_ld_post_pre_base(env, ctx);


    case OPCM_32_BO_ADDRMODE_LD_BITREVERSE_CIRCULAR:

        decode_bo_addrmode_ld_bitreverse_circular(env, ctx);


    case OPCM_32_BO_ADDRMODE_STCTX_POST_PRE_BASE:

        decode_bo_addrmode_stctx_post_pre_base(env, ctx);


    case OPCM_32_BO_ADDRMODE_LDMST_BITREVERSE_CIRCULAR:

        decode_bo_addrmode_ldmst_bitreverse_circular(env, ctx);


/* BOL-format */

    case OPC1_32_BOL_LD_A_LONGOFF:

    case OPC1_32_BOL_LD_W_LONGOFF:

    case OPC1_32_BOL_LEA_LONGOFF:

    case OPC1_32_BOL_ST_W_LONGOFF:

    case OPC1_32_BOL_ST_A_LONGOFF:

        decode_bol_opc(env, ctx, op1);


/* BRC Format */

    case OPCM_32_BRC_EQ_NEQ:

    case OPCM_32_BRC_GE:

    case OPCM_32_BRC_JLT:

    case OPCM_32_BRC_JNE:

        const4 = MASK_OP_BRC_CONST4_SEXT(ctx->opcode);

        address = MASK_OP_BRC_DISP15_SEXT(ctx->opcode);

        r1 = MASK_OP_BRC_S1(ctx->opcode);

        gen_compute_branch(ctx, op1, r1, 0, const4, address);


/* BRN Format */

    case OPCM_32_BRN_JTT:

        address = MASK_OP_BRN_DISP15_SEXT(ctx->opcode);

        r1 = MASK_OP_BRN_S1(ctx->opcode);

        gen_compute_branch(ctx, op1, r1, 0, 0, address);


/* BRR Format */

    case OPCM_32_BRR_EQ_NEQ:

    case OPCM_32_BRR_ADDR_EQ_NEQ:

    case OPCM_32_BRR_GE:

    case OPCM_32_BRR_JLT:

    case OPCM_32_BRR_JNE:

    case OPCM_32_BRR_JNZ:

    case OPCM_32_BRR_LOOP:

        address = MASK_OP_BRR_DISP15_SEXT(ctx->opcode);

        r2 = MASK_OP_BRR_S2(ctx->opcode);

        r1 = MASK_OP_BRR_S1(ctx->opcode);

        gen_compute_branch(ctx, op1, r1, r2, 0, address);


/* RC Format */

    case OPCM_32_RC_LOGICAL_SHIFT:

        decode_rc_logical_shift(env, ctx);


    case OPCM_32_RC_ACCUMULATOR:

        decode_rc_accumulator(env, ctx);


    case OPCM_32_RC_SERVICEROUTINE:

        decode_rc_serviceroutine(env, ctx);


    case OPCM_32_RC_MUL:

        decode_rc_mul(env, ctx);


/* RCPW Format */

    case OPCM_32_RCPW_MASK_INSERT:

        decode_rcpw_insert(env, ctx);


/* RCRR Format */

    case OPC1_32_RCRR_INSERT:

        r1 = MASK_OP_RCRR_S1(ctx->opcode);

        r2 = MASK_OP_RCRR_S3(ctx->opcode);

        r3 = MASK_OP_RCRR_D(ctx->opcode);

        const16 = MASK_OP_RCRR_CONST4(ctx->opcode);

        temp = tcg_const_i32(const16);

        temp2 = tcg_temp_new(); /* width*/

        temp3 = tcg_temp_new(); /* pos */



        tcg_gen_andi_tl(temp2, cpu_gpr_d[r3+1], 0x1f);

        tcg_gen_andi_tl(temp3, cpu_gpr_d[r3], 0x1f);



        gen_insert(cpu_gpr_d[r2], cpu_gpr_d[r1], temp, temp2, temp3);



        tcg_temp_free(temp);

        tcg_temp_free(temp2);

        tcg_temp_free(temp3);


/* RCRW Format */

    case OPCM_32_RCRW_MASK_INSERT:

        decode_rcrw_insert(env, ctx);


/* RCR Format */

    case OPCM_32_RCR_COND_SELECT:

        decode_rcr_cond_select(env, ctx);


    case OPCM_32_RCR_MADD:

        decode_rcr_madd(env, ctx);


    case OPCM_32_RCR_MSUB:

        decode_rcr_msub(env, ctx);


/* RLC Format */

    case OPC1_32_RLC_ADDI:

    case OPC1_32_RLC_ADDIH:

    case OPC1_32_RLC_ADDIH_A:

    case OPC1_32_RLC_MFCR:

    case OPC1_32_RLC_MOV:

    case OPC1_32_RLC_MOV_64:

    case OPC1_32_RLC_MOV_U:

    case OPC1_32_RLC_MOV_H:

    case OPC1_32_RLC_MOVH_A:

    case OPC1_32_RLC_MTCR:

        decode_rlc_opc(env, ctx, op1);






    }

}