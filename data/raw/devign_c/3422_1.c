static void dec_float(DisasContext *dc, uint32_t insn)

{

    uint32_t op0;

    uint32_t ra, rb, rd;

    op0 = extract32(insn, 0, 8);

    ra = extract32(insn, 16, 5);

    rb = extract32(insn, 11, 5);

    rd = extract32(insn, 21, 5);



    switch (op0) {

    case 0x00:    /* lf.add.s */

        LOG_DIS("lf.add.s r%d, r%d, r%d\n", rd, ra, rb);

        gen_helper_float_add_s(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x01:    /* lf.sub.s */

        LOG_DIS("lf.sub.s r%d, r%d, r%d\n", rd, ra, rb);

        gen_helper_float_sub_s(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;





    case 0x02:    /* lf.mul.s */

        LOG_DIS("lf.mul.s r%d, r%d, r%d\n", rd, ra, rb);

        if (ra != 0 && rb != 0) {

            gen_helper_float_mul_s(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        } else {

            tcg_gen_ori_tl(fpcsr, fpcsr, FPCSR_ZF);

            tcg_gen_movi_i32(cpu_R[rd], 0x0);

        }

        break;



    case 0x03:    /* lf.div.s */

        LOG_DIS("lf.div.s r%d, r%d, r%d\n", rd, ra, rb);

        gen_helper_float_div_s(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x04:    /* lf.itof.s */

        LOG_DIS("lf.itof r%d, r%d\n", rd, ra);

        gen_helper_itofs(cpu_R[rd], cpu_env, cpu_R[ra]);

        break;



    case 0x05:    /* lf.ftoi.s */

        LOG_DIS("lf.ftoi r%d, r%d\n", rd, ra);

        gen_helper_ftois(cpu_R[rd], cpu_env, cpu_R[ra]);

        break;



    case 0x06:    /* lf.rem.s */

        LOG_DIS("lf.rem.s r%d, r%d, r%d\n", rd, ra, rb);

        gen_helper_float_rem_s(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x07:    /* lf.madd.s */

        LOG_DIS("lf.madd.s r%d, r%d, r%d\n", rd, ra, rb);

        gen_helper_float_madd_s(cpu_R[rd], cpu_env, cpu_R[rd],

                                cpu_R[ra], cpu_R[rb]);

        break;



    case 0x08:    /* lf.sfeq.s */

        LOG_DIS("lf.sfeq.s r%d, r%d\n", ra, rb);

        gen_helper_float_eq_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x09:    /* lf.sfne.s */

        LOG_DIS("lf.sfne.s r%d, r%d\n", ra, rb);

        gen_helper_float_ne_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x0a:    /* lf.sfgt.s */

        LOG_DIS("lf.sfgt.s r%d, r%d\n", ra, rb);

        gen_helper_float_gt_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x0b:    /* lf.sfge.s */

        LOG_DIS("lf.sfge.s r%d, r%d\n", ra, rb);

        gen_helper_float_ge_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x0c:    /* lf.sflt.s */

        LOG_DIS("lf.sflt.s r%d, r%d\n", ra, rb);

        gen_helper_float_lt_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x0d:    /* lf.sfle.s */

        LOG_DIS("lf.sfle.s r%d, r%d\n", ra, rb);

        gen_helper_float_le_s(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



/* not used yet, open it when we need or64.  */

/*#ifdef TARGET_OPENRISC64

    case 0x10:     lf.add.d

        LOG_DIS("lf.add.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        gen_helper_float_add_d(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x11:     lf.sub.d

        LOG_DIS("lf.sub.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        gen_helper_float_sub_d(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x12:     lf.mul.d

        LOG_DIS("lf.mul.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        if (ra != 0 && rb != 0) {

            gen_helper_float_mul_d(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        } else {

            tcg_gen_ori_tl(fpcsr, fpcsr, FPCSR_ZF);

            tcg_gen_movi_i64(cpu_R[rd], 0x0);

        }

        break;



    case 0x13:     lf.div.d

        LOG_DIS("lf.div.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        gen_helper_float_div_d(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x14:     lf.itof.d

        LOG_DIS("lf.itof r%d, r%d\n", rd, ra);

        check_of64s(dc);

        gen_helper_itofd(cpu_R[rd], cpu_env, cpu_R[ra]);

        break;



    case 0x15:     lf.ftoi.d

        LOG_DIS("lf.ftoi r%d, r%d\n", rd, ra);

        check_of64s(dc);

        gen_helper_ftoid(cpu_R[rd], cpu_env, cpu_R[ra]);

        break;



    case 0x16:     lf.rem.d

        LOG_DIS("lf.rem.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        gen_helper_float_rem_d(cpu_R[rd], cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x17:     lf.madd.d

        LOG_DIS("lf.madd.d r%d, r%d, r%d\n", rd, ra, rb);

        check_of64s(dc);

        gen_helper_float_madd_d(cpu_R[rd], cpu_env, cpu_R[rd],

                                cpu_R[ra], cpu_R[rb]);

        break;



    case 0x18:     lf.sfeq.d

        LOG_DIS("lf.sfeq.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_eq_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x1a:     lf.sfgt.d

        LOG_DIS("lf.sfgt.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_gt_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x1b:     lf.sfge.d

        LOG_DIS("lf.sfge.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_ge_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x19:     lf.sfne.d

        LOG_DIS("lf.sfne.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_ne_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x1c:     lf.sflt.d

        LOG_DIS("lf.sflt.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_lt_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;



    case 0x1d:     lf.sfle.d

        LOG_DIS("lf.sfle.d r%d, r%d\n", ra, rb);

        check_of64s(dc);

        gen_helper_float_le_d(cpu_sr_f, cpu_env, cpu_R[ra], cpu_R[rb]);

        break;

#endif*/



    default:

        gen_illegal_exception(dc);

        break;

    }

}
