void tcg_target_qemu_prologue(TCGContext *s)

{

    /* stmdb sp!, { r9 - r11, lr } */

    tcg_out32(s, (COND_AL << 28) | 0x092d4e00);



    tcg_out_bx(s, COND_AL, TCG_REG_R0);

    tb_ret_addr = s->code_ptr;



    /* ldmia sp!, { r9 - r11, pc } */

    tcg_out32(s, (COND_AL << 28) | 0x08bd8e00);

}
