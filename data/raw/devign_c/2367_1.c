static void tcg_target_qemu_prologue(TCGContext *s)

{

    int i, frame_size;



    /* reserve some stack space */

    frame_size = ARRAY_SIZE(tcg_target_callee_save_regs) * 4

                 + TCG_STATIC_CALL_ARGS_SIZE;

    frame_size = (frame_size + TCG_TARGET_STACK_ALIGN - 1) &

                 ~(TCG_TARGET_STACK_ALIGN - 1);



    /* TB prologue */

    tcg_out_addi(s, TCG_REG_SP, -frame_size);

    for(i = 0 ; i < ARRAY_SIZE(tcg_target_callee_save_regs) ; i++) {

        tcg_out_st(s, TCG_TYPE_I32, tcg_target_callee_save_regs[i],

                   TCG_REG_SP, TCG_STATIC_CALL_ARGS_SIZE + i * 4);

    }



    /* Call generated code */

    tcg_out_opc_reg(s, OPC_JR, 0, tcg_target_call_iarg_regs[1]), 0);

    tcg_out_mov(s, TCG_TYPE_PTR, TCG_AREG0, tcg_target_call_iarg_regs[0]);

    tb_ret_addr = s->code_ptr;



    /* TB epilogue */

    for(i = 0 ; i < ARRAY_SIZE(tcg_target_callee_save_regs) ; i++) {

        tcg_out_ld(s, TCG_TYPE_I32, tcg_target_callee_save_regs[i],

                   TCG_REG_SP, TCG_STATIC_CALL_ARGS_SIZE + i * 4);

    }



    tcg_out_opc_reg(s, OPC_JR, 0, TCG_REG_RA, 0);

    tcg_out_addi(s, TCG_REG_SP, frame_size);

}
