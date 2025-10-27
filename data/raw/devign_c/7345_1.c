static double get_volume(CompandContext *s, double in_lin)

{

    CompandSegment *cs;

    double in_log, out_log;

    int i;



    if (in_lin < s->in_min_lin)

        return s->out_min_lin;



    in_log = log(in_lin);



    for (i = 1;; i++)

        if (in_log <= s->segments[i + 1].x)

            break;



    cs = &s->segments[i];

    in_log -= cs->x;

    out_log = cs->y + in_log * (cs->a * in_log + cs->b);



    return exp(out_log);

}
