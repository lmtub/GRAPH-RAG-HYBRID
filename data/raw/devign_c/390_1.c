static void gen_exception_return(DisasContext *s, TCGv pc)

{

    TCGv tmp;

    store_reg(s, 15, pc);

    tmp = load_cpu_field(spsr);

    gen_set_cpsr(tmp, 0xffffffff);

    dead_tmp(tmp);

    s->is_jmp = DISAS_UPDATE;

}
