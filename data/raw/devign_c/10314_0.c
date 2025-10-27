static always_inline void gen_farith3 (void *helper,

                                       int ra, int rb, int rc)

{

    if (unlikely(rc == 31))

        return;



    if (ra != 31) {

        if (rb != 31)

            tcg_gen_helper_1_2(helper, cpu_fir[rc], cpu_fir[ra], cpu_fir[rb]);

        else {

            TCGv tmp = tcg_const_i64(0);

            tcg_gen_helper_1_2(helper, cpu_fir[rc], cpu_fir[ra], tmp);

            tcg_temp_free(tmp);

        }

    } else {

        TCGv tmp = tcg_const_i64(0);

        if (rb != 31)

            tcg_gen_helper_1_2(helper, cpu_fir[rc], tmp, cpu_fir[rb]);

        else

            tcg_gen_helper_1_2(helper, cpu_fir[rc], tmp, tmp);

        tcg_temp_free(tmp);

    }

}
