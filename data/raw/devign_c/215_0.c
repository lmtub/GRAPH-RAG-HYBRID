static void handle_2misc_fcmp_zero(DisasContext *s, int opcode,

                                   bool is_scalar, bool is_u, bool is_q,

                                   int size, int rn, int rd)

{

    bool is_double = (size == 3);

    TCGv_ptr fpst = get_fpstatus_ptr();



    if (is_double) {

        TCGv_i64 tcg_op = tcg_temp_new_i64();

        TCGv_i64 tcg_zero = tcg_const_i64(0);

        TCGv_i64 tcg_res = tcg_temp_new_i64();

        NeonGenTwoDoubleOPFn *genfn;

        bool swap = false;

        int pass;



        switch (opcode) {

        case 0x2e: /* FCMLT (zero) */

            swap = true;

            /* fallthrough */

        case 0x2c: /* FCMGT (zero) */

            genfn = gen_helper_neon_cgt_f64;

            break;

        case 0x2d: /* FCMEQ (zero) */

            genfn = gen_helper_neon_ceq_f64;

            break;

        case 0x6d: /* FCMLE (zero) */

            swap = true;

            /* fall through */

        case 0x6c: /* FCMGE (zero) */

            genfn = gen_helper_neon_cge_f64;

            break;

        default:

            g_assert_not_reached();

        }



        for (pass = 0; pass < (is_scalar ? 1 : 2); pass++) {

            read_vec_element(s, tcg_op, rn, pass, MO_64);

            if (swap) {

                genfn(tcg_res, tcg_zero, tcg_op, fpst);

            } else {

                genfn(tcg_res, tcg_op, tcg_zero, fpst);

            }

            write_vec_element(s, tcg_res, rd, pass, MO_64);

        }

        if (is_scalar) {

            clear_vec_high(s, rd);

        }



        tcg_temp_free_i64(tcg_res);

        tcg_temp_free_i64(tcg_zero);

        tcg_temp_free_i64(tcg_op);

    } else {

        TCGv_i32 tcg_op = tcg_temp_new_i32();

        TCGv_i32 tcg_zero = tcg_const_i32(0);

        TCGv_i32 tcg_res = tcg_temp_new_i32();

        NeonGenTwoSingleOPFn *genfn;

        bool swap = false;

        int pass, maxpasses;



        switch (opcode) {

        case 0x2e: /* FCMLT (zero) */

            swap = true;

            /* fall through */

        case 0x2c: /* FCMGT (zero) */

            genfn = gen_helper_neon_cgt_f32;

            break;

        case 0x2d: /* FCMEQ (zero) */

            genfn = gen_helper_neon_ceq_f32;

            break;

        case 0x6d: /* FCMLE (zero) */

            swap = true;

            /* fall through */

        case 0x6c: /* FCMGE (zero) */

            genfn = gen_helper_neon_cge_f32;

            break;

        default:

            g_assert_not_reached();

        }



        if (is_scalar) {

            maxpasses = 1;

        } else {

            maxpasses = is_q ? 4 : 2;

        }



        for (pass = 0; pass < maxpasses; pass++) {

            read_vec_element_i32(s, tcg_op, rn, pass, MO_32);

            if (swap) {

                genfn(tcg_res, tcg_zero, tcg_op, fpst);

            } else {

                genfn(tcg_res, tcg_op, tcg_zero, fpst);

            }

            if (is_scalar) {

                write_fp_sreg(s, rd, tcg_res);

            } else {

                write_vec_element_i32(s, tcg_res, rd, pass, MO_32);

            }

        }

        tcg_temp_free_i32(tcg_res);

        tcg_temp_free_i32(tcg_zero);

        tcg_temp_free_i32(tcg_op);

        if (!is_q && !is_scalar) {

            clear_vec_high(s, rd);

        }

    }



    tcg_temp_free_ptr(fpst);

}
