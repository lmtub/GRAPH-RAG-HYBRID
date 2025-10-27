static void handle_fmov(DisasContext *s, int rd, int rn, int type, bool itof)

{

    /* FMOV: gpr to or from float, double, or top half of quad fp reg,

     * without conversion.

     */



    if (itof) {

        TCGv_i64 tcg_rn = cpu_reg(s, rn);



        switch (type) {

        case 0:

        {

            /* 32 bit */

            TCGv_i64 tmp = tcg_temp_new_i64();

            tcg_gen_ext32u_i64(tmp, tcg_rn);

            tcg_gen_st_i64(tmp, cpu_env, fp_reg_offset(rd, MO_64));

            tcg_gen_movi_i64(tmp, 0);

            tcg_gen_st_i64(tmp, cpu_env, fp_reg_hi_offset(rd));

            tcg_temp_free_i64(tmp);

            break;

        }

        case 1:

        {

            /* 64 bit */

            TCGv_i64 tmp = tcg_const_i64(0);

            tcg_gen_st_i64(tcg_rn, cpu_env, fp_reg_offset(rd, MO_64));

            tcg_gen_st_i64(tmp, cpu_env, fp_reg_hi_offset(rd));

            tcg_temp_free_i64(tmp);

            break;

        }

        case 2:

            /* 64 bit to top half. */

            tcg_gen_st_i64(tcg_rn, cpu_env, fp_reg_hi_offset(rd));

            break;

        }

    } else {

        TCGv_i64 tcg_rd = cpu_reg(s, rd);



        switch (type) {

        case 0:

            /* 32 bit */

            tcg_gen_ld32u_i64(tcg_rd, cpu_env, fp_reg_offset(rn, MO_32));

            break;

        case 1:

            /* 64 bit */

            tcg_gen_ld_i64(tcg_rd, cpu_env, fp_reg_offset(rn, MO_64));

            break;

        case 2:

            /* 64 bits from top half */

            tcg_gen_ld_i64(tcg_rd, cpu_env, fp_reg_hi_offset(rn));

            break;

        }

    }

}
