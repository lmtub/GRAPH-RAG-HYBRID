void helper_fcmp_eq_DT(CPUSH4State *env, float64 t0, float64 t1)

{

    int relation;



    set_float_exception_flags(0, &env->fp_status);

    relation = float64_compare(t0, t1, &env->fp_status);

    if (unlikely(relation == float_relation_unordered)) {

        update_fpscr(env, GETPC());

    } else {

        env->sr_t = (relation == float_relation_equal);

    }

}
