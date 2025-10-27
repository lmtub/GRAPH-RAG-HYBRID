AVEvalExpr * ff_parse(const char *s, const char * const *const_name,

               double (**func1)(void *, double), const char **func1_name,

               double (**func2)(void *, double, double), const char **func2_name,

               const char **error){

    Parser p;

    AVEvalExpr * e;

    char w[strlen(s) + 1], * wp = w;



    while (*s)

        if (!isspace(*s++)) *wp++ = s[-1];

    *wp++ = 0;



    p.stack_index=100;

    p.s= w;

    p.const_name = const_name;

    p.func1      = func1;

    p.func1_name = func1_name;

    p.func2      = func2;

    p.func2_name = func2_name;

    p.error= error;



    e = parse_expr(&p);

    if (!verify_expr(e)) {

        ff_eval_free(e);

        return NULL;

    }

    return e;

}
