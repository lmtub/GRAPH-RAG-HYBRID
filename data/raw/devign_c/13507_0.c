void tcg_dump_ops(TCGContext *s)

{

    char buf[128];

    TCGOp *op;

    int oi;



    for (oi = s->gen_first_op_idx; oi >= 0; oi = op->next) {

        int i, k, nb_oargs, nb_iargs, nb_cargs;

        const TCGOpDef *def;

        const TCGArg *args;

        TCGOpcode c;



        op = &s->gen_op_buf[oi];

        c = op->opc;

        def = &tcg_op_defs[c];

        args = &s->gen_opparam_buf[op->args];



        if (c == INDEX_op_insn_start) {

            qemu_log("%s ----", oi != s->gen_first_op_idx ? "\n" : "");



            for (i = 0; i < TARGET_INSN_START_WORDS; ++i) {

                target_ulong a;

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

                a = ((target_ulong)args[i * 2 + 1] << 32) | args[i * 2];

#else

                a = args[i];

#endif

                qemu_log(" " TARGET_FMT_lx, a);

            }

        } else if (c == INDEX_op_call) {

            /* variable number of arguments */

            nb_oargs = op->callo;

            nb_iargs = op->calli;

            nb_cargs = def->nb_cargs;



            /* function name, flags, out args */

            qemu_log(" %s %s,$0x%" TCG_PRIlx ",$%d", def->name,

                     tcg_find_helper(s, args[nb_oargs + nb_iargs]),

                     args[nb_oargs + nb_iargs + 1], nb_oargs);

            for (i = 0; i < nb_oargs; i++) {

                qemu_log(",%s", tcg_get_arg_str_idx(s, buf, sizeof(buf),

                                                   args[i]));

            }

            for (i = 0; i < nb_iargs; i++) {

                TCGArg arg = args[nb_oargs + i];

                const char *t = "<dummy>";

                if (arg != TCG_CALL_DUMMY_ARG) {

                    t = tcg_get_arg_str_idx(s, buf, sizeof(buf), arg);

                }

                qemu_log(",%s", t);

            }

        } else {

            qemu_log(" %s ", def->name);



            nb_oargs = def->nb_oargs;

            nb_iargs = def->nb_iargs;

            nb_cargs = def->nb_cargs;



            k = 0;

            for (i = 0; i < nb_oargs; i++) {

                if (k != 0) {

                    qemu_log(",");

                }

                qemu_log("%s", tcg_get_arg_str_idx(s, buf, sizeof(buf),

                                                   args[k++]));

            }

            for (i = 0; i < nb_iargs; i++) {

                if (k != 0) {

                    qemu_log(",");

                }

                qemu_log("%s", tcg_get_arg_str_idx(s, buf, sizeof(buf),

                                                   args[k++]));

            }

            switch (c) {

            case INDEX_op_brcond_i32:

            case INDEX_op_setcond_i32:

            case INDEX_op_movcond_i32:

            case INDEX_op_brcond2_i32:

            case INDEX_op_setcond2_i32:

            case INDEX_op_brcond_i64:

            case INDEX_op_setcond_i64:

            case INDEX_op_movcond_i64:

                if (args[k] < ARRAY_SIZE(cond_name) && cond_name[args[k]]) {

                    qemu_log(",%s", cond_name[args[k++]]);

                } else {

                    qemu_log(",$0x%" TCG_PRIlx, args[k++]);

                }

                i = 1;

                break;

            case INDEX_op_qemu_ld_i32:

            case INDEX_op_qemu_st_i32:

            case INDEX_op_qemu_ld_i64:

            case INDEX_op_qemu_st_i64:

                {

                    TCGMemOpIdx oi = args[k++];

                    TCGMemOp op = get_memop(oi);

                    unsigned ix = get_mmuidx(oi);



                    if (op & ~(MO_AMASK | MO_BSWAP | MO_SSIZE)) {

                        qemu_log(",$0x%x,%u", op, ix);

                    } else {

                        const char *s_al = "", *s_op;

                        if (op & MO_AMASK) {

                            if ((op & MO_AMASK) == MO_ALIGN) {

                                s_al = "al+";

                            } else {

                                s_al = "un+";

                            }

                        }

                        s_op = ldst_name[op & (MO_BSWAP | MO_SSIZE)];

                        qemu_log(",%s%s,%u", s_al, s_op, ix);

                    }

                    i = 1;

                }

                break;

            default:

                i = 0;

                break;

            }

            switch (c) {

            case INDEX_op_set_label:

            case INDEX_op_br:

            case INDEX_op_brcond_i32:

            case INDEX_op_brcond_i64:

            case INDEX_op_brcond2_i32:

                qemu_log("%s$L%d", k ? "," : "", arg_label(args[k])->id);

                i++, k++;

                break;

            default:

                break;

            }

            for (; i < nb_cargs; i++, k++) {

                qemu_log("%s$0x%" TCG_PRIlx, k ? "," : "", args[k]);

            }

        }

        qemu_log("\n");

    }

}
