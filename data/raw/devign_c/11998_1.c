static void gen_exception_return(DisasContext *s, TCGv_i32 pc)

{

    TCGv_i32 tmp;

    store_reg(s, 15, pc);

    tmp = load_cpu_field(spsr);

    gen_set_cpsr(tmp, CPSR_ERET_MASK);

    tcg_temp_free_i32(tmp);

    s->is_jmp = DISAS_UPDATE;

}
