static void tcg_out_setcond(TCGContext *s, TCGType type, TCGCond cond,

                            TCGArg arg0, TCGArg arg1, TCGArg arg2,

                            int const_arg2)

{

    int crop, sh;



    assert(TCG_TARGET_REG_BITS == 64 || type == TCG_TYPE_I32);



    /* Ignore high bits of a potential constant arg2.  */

    if (type == TCG_TYPE_I32) {

        arg2 = (uint32_t)arg2;

    }



    /* Handle common and trivial cases before handling anything else.  */

    if (arg2 == 0) {

        switch (cond) {

        case TCG_COND_EQ:

            tcg_out_setcond_eq0(s, type, arg0, arg1);

            return;

        case TCG_COND_NE:

            if (TCG_TARGET_REG_BITS == 64 && type == TCG_TYPE_I32) {

                tcg_out_ext32u(s, TCG_REG_R0, arg1);

                arg1 = TCG_REG_R0;

            }

            tcg_out_setcond_ne0(s, arg0, arg1);

            return;

        case TCG_COND_GE:

            tcg_out32(s, NOR | SAB(arg1, arg0, arg1));

            arg1 = arg0;

            /* FALLTHRU */

        case TCG_COND_LT:

            /* Extract the sign bit.  */

            if (type == TCG_TYPE_I32) {

                tcg_out_shri32(s, arg0, arg1, 31);

            } else {

                tcg_out_shri64(s, arg0, arg1, 63);

            }

            return;

        default:

            break;

        }

    }



    /* If we have ISEL, we can implement everything with 3 or 4 insns.

       All other cases below are also at least 3 insns, so speed up the

       code generator by not considering them and always using ISEL.  */

    if (HAVE_ISEL) {

        int isel, tab;



        tcg_out_cmp(s, cond, arg1, arg2, const_arg2, 7, type);



        isel = tcg_to_isel[cond];



        tcg_out_movi(s, type, arg0, 1);

        if (isel & 1) {

            /* arg0 = (bc ? 0 : 1) */

            tab = TAB(arg0, 0, arg0);

            isel &= ~1;

        } else {

            /* arg0 = (bc ? 1 : 0) */

            tcg_out_movi(s, type, TCG_REG_R0, 0);

            tab = TAB(arg0, arg0, TCG_REG_R0);

        }

        tcg_out32(s, isel | tab);

        return;

    }



    switch (cond) {

    case TCG_COND_EQ:

        arg1 = tcg_gen_setcond_xor(s, arg1, arg2, const_arg2);

        tcg_out_setcond_eq0(s, type, arg0, arg1);

        return;



    case TCG_COND_NE:

        arg1 = tcg_gen_setcond_xor(s, arg1, arg2, const_arg2);

        /* Discard the high bits only once, rather than both inputs.  */

        if (TCG_TARGET_REG_BITS == 64 && type == TCG_TYPE_I32) {

            tcg_out_ext32u(s, TCG_REG_R0, arg1);

            arg1 = TCG_REG_R0;

        }

        tcg_out_setcond_ne0(s, arg0, arg1);

        return;



    case TCG_COND_GT:

    case TCG_COND_GTU:

        sh = 30;

        crop = 0;

        goto crtest;



    case TCG_COND_LT:

    case TCG_COND_LTU:

        sh = 29;

        crop = 0;

        goto crtest;



    case TCG_COND_GE:

    case TCG_COND_GEU:

        sh = 31;

        crop = CRNOR | BT(7, CR_EQ) | BA(7, CR_LT) | BB(7, CR_LT);

        goto crtest;



    case TCG_COND_LE:

    case TCG_COND_LEU:

        sh = 31;

        crop = CRNOR | BT(7, CR_EQ) | BA(7, CR_GT) | BB(7, CR_GT);

    crtest:

        tcg_out_cmp(s, cond, arg1, arg2, const_arg2, 7, type);

        if (crop) {

            tcg_out32(s, crop);

        }

        tcg_out32(s, MFOCRF | RT(TCG_REG_R0) | FXM(7));

        tcg_out_rlw(s, RLWINM, arg0, TCG_REG_R0, sh, 31, 31);

        break;



    default:

        tcg_abort();

    }

}
