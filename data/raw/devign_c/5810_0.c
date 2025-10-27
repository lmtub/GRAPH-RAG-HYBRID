static void gen_exception_insn(DisasContext *s, int offset, int excp)

{

    gen_a64_set_pc_im(s->pc - offset);

    gen_exception(excp);

    s->is_jmp = DISAS_JUMP;

}
