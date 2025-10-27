static always_inline void gen_arith3 (void *helper,

                                      int ra, int rb, int rc,

                                      int islit, uint8_t lit)

{

    if (unlikely(rc == 31))

        return;



    if (ra != 31) {

        if (islit) {

            TCGv tmp = tcg_const_i64(lit);

            tcg_gen_helper_1_2(helper, cpu_ir[rc], cpu_ir[ra], tmp);

            tcg_temp_free(tmp);

        } else

            tcg_gen_helper_1_2(helper, cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

    } else {

        TCGv tmp1 = tcg_const_i64(0);

        if (islit) {

            TCGv tmp2 = tcg_const_i64(lit);

            tcg_gen_helper_1_2(helper, cpu_ir[rc], tmp1, tmp2);

            tcg_temp_free(tmp2);

        } else

            tcg_gen_helper_1_2(helper, cpu_ir[rc], tmp1, cpu_ir[rb]);

        tcg_temp_free(tmp1);

    }

}
