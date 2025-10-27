int av_expr_parse(AVExpr **expr, const char *s,

                  const char * const *const_names,

                  const char * const *func1_names, double (* const *funcs1)(void *, double),

                  const char * const *func2_names, double (* const *funcs2)(void *, double, double),

                  int log_offset, void *log_ctx)

{

    Parser p;

    AVExpr *e = NULL;

    char *w = av_malloc(strlen(s) + 1);

    char *wp = w;

    const char *s0 = s;

    int ret = 0;



    if (!w)

        return AVERROR(ENOMEM);



    while (*s)

        if (!isspace(*s++)) *wp++ = s[-1];

    *wp++ = 0;



    p.class      = &class;

    p.stack_index=100;

    p.s= w;

    p.const_names = const_names;

    p.funcs1      = funcs1;

    p.func1_names = func1_names;

    p.funcs2      = funcs2;

    p.func2_names = func2_names;

    p.log_offset = log_offset;

    p.log_ctx    = log_ctx;



    if ((ret = parse_expr(&e, &p)) < 0)

        goto end;

    if (*p.s) {

        av_log(&p, AV_LOG_ERROR, "Invalid chars '%s' at the end of expression '%s'\n", p.s, s0);

        ret = AVERROR(EINVAL);

        goto end;

    }

    if (!verify_expr(e)) {

        av_expr_free(e);

        ret = AVERROR(EINVAL);

        goto end;

    }

    *expr = e;

end:

    av_free(w);

    return ret;

}
