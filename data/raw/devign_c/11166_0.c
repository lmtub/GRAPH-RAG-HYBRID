static void gen_exception_insn(DisasContext *s, int offset, int excp,

                               int syn, uint32_t target_el)

{

    gen_set_condexec(s);

    gen_set_pc_im(s, s->pc - offset);

    gen_exception(excp, syn, target_el);

    s->is_jmp = DISAS_JUMP;

}
