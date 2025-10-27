double av_expr_eval(AVExpr *e, const double *const_values, void *opaque)

{

    Parser p;



    p.const_values = const_values;

    p.opaque     = opaque;

    return eval_expr(&p, e);

}
