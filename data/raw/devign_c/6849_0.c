static void gen_exception_internal_insn(DisasContext *s, int offset, int excp)

{

    gen_set_condexec(s);

    gen_set_pc_im(s, s->pc - offset);

    gen_exception_internal(excp);

    s->is_jmp = DISAS_JUMP;

}
