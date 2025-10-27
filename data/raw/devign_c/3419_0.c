static double eval_expr(Parser * p, AVEvalExpr * e) {

    switch (e->type) {

        case e_value:  return e->value;

        case e_const:  return e->value * p->const_value[e->a.const_index];

        case e_func0:  return e->value * e->a.func0(eval_expr(p, e->param[0]));

        case e_func1:  return e->value * e->a.func1(p->opaque, eval_expr(p, e->param[0]));

        case e_func2:  return e->value * e->a.func2(p->opaque, eval_expr(p, e->param[0]), eval_expr(p, e->param[1]));

        case e_squish: return 1/(1+exp(4*eval_expr(p, e->param[0])));

        case e_gauss: { double d = eval_expr(p, e->param[0]); return exp(-d*d/2)/sqrt(2*M_PI); }

        case e_ld:     return e->value * p->var[clip(eval_expr(p, e->param[0]), 0, VARS-1)];

        case e_while: {

            double d;

            while(eval_expr(p, e->param[0]))

                d=eval_expr(p, e->param[1]);

            return d;

        }

        default: {

            double d = eval_expr(p, e->param[0]);

            double d2 = eval_expr(p, e->param[1]);

            switch (e->type) {

                case e_mod: return e->value * (d - floor(d/d2)*d2);

                case e_max: return e->value * (d >  d2 ?   d : d2);

                case e_min: return e->value * (d <  d2 ?   d : d2);

                case e_eq:  return e->value * (d == d2 ? 1.0 : 0.0);

                case e_gt:  return e->value * (d >  d2 ? 1.0 : 0.0);

                case e_gte: return e->value * (d >= d2 ? 1.0 : 0.0);

                case e_pow: return e->value * pow(d, d2);

                case e_mul: return e->value * (d * d2);

                case e_div: return e->value * (d / d2);

                case e_add: return e->value * (d + d2);

                case e_last:return d2;

                case e_st : return e->value * (p->var[clip(d, 0, VARS-1)]= d2);

            }

        }

    }

    return NAN;

}
