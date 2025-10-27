static double eval_expr(Parser *p, AVExpr *e)

{

    switch (e->type) {

        case e_value:  return e->value;

        case e_const:  return e->value * p->const_values[e->a.const_index];

        case e_func0:  return e->value * e->a.func0(eval_expr(p, e->param[0]));

        case e_func1:  return e->value * e->a.func1(p->opaque, eval_expr(p, e->param[0]));

        case e_func2:  return e->value * e->a.func2(p->opaque, eval_expr(p, e->param[0]), eval_expr(p, e->param[1]));

        case e_squish: return 1/(1+exp(4*eval_expr(p, e->param[0])));

        case e_gauss: { double d = eval_expr(p, e->param[0]); return exp(-d*d/2)/sqrt(2*M_PI); }

        case e_ld:     return e->value * p->var[av_clip(eval_expr(p, e->param[0]), 0, VARS-1)];

        case e_isnan:  return e->value * !!isnan(eval_expr(p, e->param[0]));

        case e_floor:  return e->value * floor(eval_expr(p, e->param[0]));

        case e_ceil :  return e->value * ceil (eval_expr(p, e->param[0]));

        case e_trunc:  return e->value * trunc(eval_expr(p, e->param[0]));

        case e_sqrt:   return e->value * sqrt (eval_expr(p, e->param[0]));

        case e_not:    return e->value * (eval_expr(p, e->param[0]) == 0);

        case e_if:     return e->value * ( eval_expr(p, e->param[0]) ? eval_expr(p, e->param[1]) : 0);

        case e_ifnot:  return e->value * (!eval_expr(p, e->param[0]) ? eval_expr(p, e->param[1]) : 0);

        case e_random:{

            int idx= av_clip(eval_expr(p, e->param[0]), 0, VARS-1);

            uint64_t r= isnan(p->var[idx]) ? 0 : p->var[idx];

            r= r*1664525+1013904223;

            p->var[idx]= r;

            return e->value * (r * (1.0/UINT64_MAX));

        }

        case e_while: {

            double d = NAN;

            while (eval_expr(p, e->param[0]))

                d=eval_expr(p, e->param[1]);

            return d;

        }

        case e_taylor: {

            double t = 1, d = 0, v;

            double x = eval_expr(p, e->param[1]);

            int id = e->param[2] ? av_clip(eval_expr(p, e->param[2]), 0, VARS-1) : 0;

            int i;

            double var0 = p->var[id];

            for(i=0; i<1000; i++) {

                double ld = d;

                p->var[id] = i;

                v = eval_expr(p, e->param[0]);

                d += t*v;

                if(ld==d && v)

                    break;

                t *= x / (i+1);

            }

            p->var[id] = var0;

            return d;

        }

        case e_root: {

            int i;

            double low = -1, high = -1, v, low_v = -DBL_MAX, high_v = DBL_MAX;

            double var0 = p->var[0];

            double x_max = eval_expr(p, e->param[1]);

            for(i=-1; i<1024; i++) {

                if(i<255) {

                    p->var[0] = av_reverse[i&255]*x_max/255;

                } else {

                    p->var[0] = x_max*pow(0.9, i-255);

                    if (i&1) p->var[0] *= -1;

                    if (i&2) p->var[0] += low;

                    else     p->var[0] += high;

                }

                v = eval_expr(p, e->param[0]);

                if (v<=0 && v>low_v) {

                    low    = p->var[0];

                    low_v  = v;

                }

                if (v>=0 && v<high_v) {

                    high   = p->var[0];

                    high_v = v;

                }

                if (low>=0 && high>=0){

                    while (1) {

                        p->var[0] = (low+high)*0.5;

                        if (low == p->var[0] || high == p->var[0])

                            break;

                        v = eval_expr(p, e->param[0]);

                        if (v<=0) low = p->var[0];

                        if (v>=0) high= p->var[0];

                        if (isnan(v)) {

                            low = high = v;

                            break;

                        }

                    }

                    break;

                }

            }

            p->var[0] = var0;

            return -low_v<high_v ? low : high;

        }

        default: {

            double d = eval_expr(p, e->param[0]);

            double d2 = eval_expr(p, e->param[1]);

            switch (e->type) {

                case e_mod: return e->value * (d - floor(d/d2)*d2);

                case e_gcd: return e->value * av_gcd(d,d2);

                case e_max: return e->value * (d >  d2 ?   d : d2);

                case e_min: return e->value * (d <  d2 ?   d : d2);

                case e_eq:  return e->value * (d == d2 ? 1.0 : 0.0);

                case e_gt:  return e->value * (d >  d2 ? 1.0 : 0.0);

                case e_gte: return e->value * (d >= d2 ? 1.0 : 0.0);

                case e_pow: return e->value * pow(d, d2);

                case e_mul: return e->value * (d * d2);

                case e_div: return e->value * (d / d2);

                case e_add: return e->value * (d + d2);

                case e_last:return e->value * d2;

                case e_st : return e->value * (p->var[av_clip(d, 0, VARS-1)]= d2);

                case e_hypot:return e->value * (sqrt(d*d + d2*d2));

            }

        }

    }

    return NAN;

}
