static void gen_check_privilege(DisasContext *dc)

{

    if (dc->cring) {

        gen_exception_cause(dc, PRIVILEGED_CAUSE);

        dc->is_jmp = DISAS_UPDATE;

    }

}
