static char *tcg_get_arg_str_idx(TCGContext *s, char *buf, int buf_size,

                                 int idx)

{

    TCGTemp *ts;



    assert(idx >= 0 && idx < s->nb_temps);

    ts = &s->temps[idx];

    assert(ts);

    if (idx < s->nb_globals) {

        pstrcpy(buf, buf_size, ts->name);

    } else {

        if (ts->temp_local) 

            snprintf(buf, buf_size, "loc%d", idx - s->nb_globals);

        else

            snprintf(buf, buf_size, "tmp%d", idx - s->nb_globals);

    }

    return buf;

}
