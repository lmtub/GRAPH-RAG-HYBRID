static void tcg_opt_gen_mov(TCGArg *gen_args, TCGArg dst, TCGArg src,

                            int nb_temps, int nb_globals)

{

        reset_temp(dst, nb_temps, nb_globals);

        assert(temps[src].state != TCG_TEMP_COPY);

        if (src >= nb_globals) {

            assert(temps[src].state != TCG_TEMP_CONST);

            if (temps[src].state != TCG_TEMP_HAS_COPY) {

                temps[src].state = TCG_TEMP_HAS_COPY;

                temps[src].next_copy = src;

                temps[src].prev_copy = src;

            }

            temps[dst].state = TCG_TEMP_COPY;

            temps[dst].val = src;

            temps[dst].next_copy = temps[src].next_copy;

            temps[dst].prev_copy = src;

            temps[temps[dst].next_copy].prev_copy = dst;

            temps[src].next_copy = dst;

        }

        gen_args[0] = dst;

        gen_args[1] = src;

}
