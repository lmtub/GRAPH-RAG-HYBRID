target_ulong helper_sdiv(target_ulong a, target_ulong b)

{

    int64_t x0;

    int32_t x1;



    x0 = (a & 0xffffffff) | ((int64_t) (env->y) << 32);

    x1 = (b & 0xffffffff);



    if (x1 == 0) {

        raise_exception(TT_DIV_ZERO);

    }



    x0 = x0 / x1;

    if ((int32_t) x0 != x0) {

        env->cc_src2 = 1;

        return x0 < 0? 0x80000000: 0x7fffffff;

    } else {

        env->cc_src2 = 0;

        return x0;

    }

}
