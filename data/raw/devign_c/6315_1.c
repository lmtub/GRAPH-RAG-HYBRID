static void gen_rfe(DisasContext *s, TCGv pc, TCGv cpsr)

{

    gen_set_cpsr(cpsr, 0xffffffff);

    dead_tmp(cpsr);

    store_reg(s, 15, pc);

    s->is_jmp = DISAS_UPDATE;

}
