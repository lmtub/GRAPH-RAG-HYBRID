static void tcg_out_brcond2(TCGContext *s, const TCGArg *args,

                            const int *const_args, int small)

{

    int label_next;

    label_next = gen_new_label();

    switch(args[4]) {

    case TCG_COND_EQ:

        tcg_out_brcond32(s, TCG_COND_NE, args[0], args[2], const_args[2],

                         label_next, 1);

        tcg_out_brcond32(s, TCG_COND_EQ, args[1], args[3], const_args[3],

                         args[5], small);

        break;

    case TCG_COND_NE:

        tcg_out_brcond32(s, TCG_COND_NE, args[0], args[2], const_args[2],

                         args[5], small);

        tcg_out_brcond32(s, TCG_COND_NE, args[1], args[3], const_args[3],

                         args[5], small);

        break;

    case TCG_COND_LT:

        tcg_out_brcond32(s, TCG_COND_LT, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_LTU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_LE:

        tcg_out_brcond32(s, TCG_COND_LT, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_LEU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_GT:

        tcg_out_brcond32(s, TCG_COND_GT, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_GTU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_GE:

        tcg_out_brcond32(s, TCG_COND_GT, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_GEU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_LTU:

        tcg_out_brcond32(s, TCG_COND_LTU, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_LTU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_LEU:

        tcg_out_brcond32(s, TCG_COND_LTU, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_LEU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_GTU:

        tcg_out_brcond32(s, TCG_COND_GTU, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_GTU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    case TCG_COND_GEU:

        tcg_out_brcond32(s, TCG_COND_GTU, args[1], args[3], const_args[3],

                         args[5], small);

        tcg_out_jxx(s, JCC_JNE, label_next, 1);

        tcg_out_brcond32(s, TCG_COND_GEU, args[0], args[2], const_args[2],

                         args[5], small);

        break;

    default:

        tcg_abort();

    }

    tcg_out_label(s, label_next, s->code_ptr);

}
