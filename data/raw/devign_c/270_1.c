static void gen_rfe(DisasContext *s, TCGv_i32 pc, TCGv_i32 cpsr)

{

    gen_set_cpsr(cpsr, CPSR_ERET_MASK);

    tcg_temp_free_i32(cpsr);

    store_reg(s, 15, pc);

    s->is_jmp = DISAS_UPDATE;

}
