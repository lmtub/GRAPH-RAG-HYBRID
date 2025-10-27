static void gen_check_cpenable(DisasContext *dc, unsigned cp)

{

    if (option_enabled(dc, XTENSA_OPTION_COPROCESSOR) &&

            !(dc->cpenable & (1 << cp))) {

        gen_exception_cause(dc, COPROCESSOR0_DISABLED + cp);

        dc->is_jmp = DISAS_UPDATE;

    }

}
