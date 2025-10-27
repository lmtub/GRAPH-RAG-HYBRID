float64 HELPER(sub_cmp_f64)(CPUState *env, float64 a, float64 b)

{

    /* ??? This may incorrectly raise exceptions.  */

    /* ??? Should flush denormals to zero.  */

    float64 res;

    res = float64_sub(a, b, &env->fp_status);

    if (float64_is_nan(res)) {

        /* +/-inf compares equal against itself, but sub returns nan.  */

        if (!float64_is_nan(a)

            && !float64_is_nan(b)) {

            res = float64_zero;

            if (float64_lt_quiet(a, res, &env->fp_status))

                res = float64_chs(res);

        }

    }

    return res;

}
