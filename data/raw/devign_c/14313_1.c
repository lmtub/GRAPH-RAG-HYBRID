DISAS_INSN(wdebug)

{

    if (IS_USER(s)) {

        gen_exception(s, s->pc - 2, EXCP_PRIVILEGE);

        return;

    }

    /* TODO: Implement wdebug.  */

    qemu_assert(0, "WDEBUG not implemented");

}
