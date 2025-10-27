static void tcg_target_qemu_prologue(TCGContext *s)

{

    int frame_size, i;



    /* Allocate space for the fixed frame marker.  */

    frame_size = -TCG_TARGET_CALL_STACK_OFFSET;

    frame_size += TCG_TARGET_STATIC_CALL_ARGS_SIZE;



    /* Allocate space for the saved registers.  */

    frame_size += ARRAY_SIZE(tcg_target_callee_save_regs) * 4;



    /* Align the allocated space.  */

    frame_size = ((frame_size + TCG_TARGET_STACK_ALIGN - 1)

                  & -TCG_TARGET_STACK_ALIGN);



    /* The return address is stored in the caller's frame.  */

    tcg_out_st(s, TCG_TYPE_PTR, TCG_REG_RP, TCG_REG_SP, -20);



    /* Allocate stack frame, saving the first register at the same time.  */

    tcg_out_ldst(s, tcg_target_callee_save_regs[0],

                 TCG_REG_SP, frame_size, INSN_STWM);



    /* Save all callee saved registers.  */

    for (i = 1; i < ARRAY_SIZE(tcg_target_callee_save_regs); i++) {

        tcg_out_st(s, TCG_TYPE_PTR, tcg_target_callee_save_regs[i],

                   TCG_REG_SP, -frame_size + i * 4);

    }



#ifdef CONFIG_USE_GUEST_BASE

    if (GUEST_BASE != 0) {

        tcg_out_movi(s, TCG_TYPE_PTR, TCG_GUEST_BASE_REG, GUEST_BASE);

        tcg_regset_set_reg(s->reserved_regs, TCG_GUEST_BASE_REG);

    }

#endif



    /* Jump to TB, and adjust R18 to be the return address.  */

    tcg_out32(s, INSN_BLE_SR4 | INSN_R2(TCG_REG_R26));

    tcg_out_mov(s, TCG_TYPE_I32, TCG_REG_R18, TCG_REG_R31);



    /* Restore callee saved registers.  */

    tcg_out_ld(s, TCG_TYPE_PTR, TCG_REG_RP, TCG_REG_SP, -frame_size - 20);

    for (i = 1; i < ARRAY_SIZE(tcg_target_callee_save_regs); i++) {

        tcg_out_ld(s, TCG_TYPE_PTR, tcg_target_callee_save_regs[i],

                   TCG_REG_SP, -frame_size + i * 4);

    }



    /* Deallocate stack frame and return.  */

    tcg_out32(s, INSN_BV | INSN_R2(TCG_REG_RP));

    tcg_out_ldst(s, tcg_target_callee_save_regs[0],

                 TCG_REG_SP, -frame_size, INSN_LDWM);

}
