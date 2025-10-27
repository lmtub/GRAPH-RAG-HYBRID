static void liveness_pass_1(TCGContext *s)

{

    int nb_globals = s->nb_globals;

    int oi, oi_prev;



    tcg_la_func_end(s);



    for (oi = s->gen_op_buf[0].prev; oi != 0; oi = oi_prev) {

        int i, nb_iargs, nb_oargs;

        TCGOpcode opc_new, opc_new2;

        bool have_opc_new2;

        TCGLifeData arg_life = 0;

        TCGTemp *arg_ts;



        TCGOp * const op = &s->gen_op_buf[oi];

        TCGOpcode opc = op->opc;

        const TCGOpDef *def = &tcg_op_defs[opc];



        oi_prev = op->prev;



        switch (opc) {

        case INDEX_op_call:

            {

                int call_flags;



                nb_oargs = op->callo;

                nb_iargs = op->calli;

                call_flags = op->args[nb_oargs + nb_iargs + 1];



                /* pure functions can be removed if their result is unused */

                if (call_flags & TCG_CALL_NO_SIDE_EFFECTS) {

                    for (i = 0; i < nb_oargs; i++) {

                        arg_ts = arg_temp(op->args[i]);

                        if (arg_ts->state != TS_DEAD) {

                            goto do_not_remove_call;

                        }

                    }

                    goto do_remove;

                } else {

                do_not_remove_call:



                    /* output args are dead */

                    for (i = 0; i < nb_oargs; i++) {

                        arg_ts = arg_temp(op->args[i]);

                        if (arg_ts->state & TS_DEAD) {

                            arg_life |= DEAD_ARG << i;

                        }

                        if (arg_ts->state & TS_MEM) {

                            arg_life |= SYNC_ARG << i;

                        }

                        arg_ts->state = TS_DEAD;

                    }



                    if (!(call_flags & (TCG_CALL_NO_WRITE_GLOBALS |

                                        TCG_CALL_NO_READ_GLOBALS))) {

                        /* globals should go back to memory */

                        for (i = 0; i < nb_globals; i++) {

                            s->temps[i].state = TS_DEAD | TS_MEM;

                        }

                    } else if (!(call_flags & TCG_CALL_NO_READ_GLOBALS)) {

                        /* globals should be synced to memory */

                        for (i = 0; i < nb_globals; i++) {

                            s->temps[i].state |= TS_MEM;

                        }

                    }



                    /* record arguments that die in this helper */

                    for (i = nb_oargs; i < nb_iargs + nb_oargs; i++) {

                        arg_ts = arg_temp(op->args[i]);

                        if (arg_ts && arg_ts->state & TS_DEAD) {

                            arg_life |= DEAD_ARG << i;

                        }

                    }

                    /* input arguments are live for preceding opcodes */

                    for (i = nb_oargs; i < nb_iargs + nb_oargs; i++) {

                        arg_ts = arg_temp(op->args[i]);

                        if (arg_ts) {

                            arg_ts->state &= ~TS_DEAD;

                        }

                    }

                }

            }

            break;

        case INDEX_op_insn_start:

            break;

        case INDEX_op_discard:

            /* mark the temporary as dead */

            arg_temp(op->args[0])->state = TS_DEAD;

            break;



        case INDEX_op_add2_i32:

            opc_new = INDEX_op_add_i32;

            goto do_addsub2;

        case INDEX_op_sub2_i32:

            opc_new = INDEX_op_sub_i32;

            goto do_addsub2;

        case INDEX_op_add2_i64:

            opc_new = INDEX_op_add_i64;

            goto do_addsub2;

        case INDEX_op_sub2_i64:

            opc_new = INDEX_op_sub_i64;

        do_addsub2:

            nb_iargs = 4;

            nb_oargs = 2;

            /* Test if the high part of the operation is dead, but not

               the low part.  The result can be optimized to a simple

               add or sub.  This happens often for x86_64 guest when the

               cpu mode is set to 32 bit.  */

            if (arg_temp(op->args[1])->state == TS_DEAD) {

                if (arg_temp(op->args[0])->state == TS_DEAD) {

                    goto do_remove;

                }

                /* Replace the opcode and adjust the args in place,

                   leaving 3 unused args at the end.  */

                op->opc = opc = opc_new;

                op->args[1] = op->args[2];

                op->args[2] = op->args[4];

                /* Fall through and mark the single-word operation live.  */

                nb_iargs = 2;

                nb_oargs = 1;

            }

            goto do_not_remove;



        case INDEX_op_mulu2_i32:

            opc_new = INDEX_op_mul_i32;

            opc_new2 = INDEX_op_muluh_i32;

            have_opc_new2 = TCG_TARGET_HAS_muluh_i32;

            goto do_mul2;

        case INDEX_op_muls2_i32:

            opc_new = INDEX_op_mul_i32;

            opc_new2 = INDEX_op_mulsh_i32;

            have_opc_new2 = TCG_TARGET_HAS_mulsh_i32;

            goto do_mul2;

        case INDEX_op_mulu2_i64:

            opc_new = INDEX_op_mul_i64;

            opc_new2 = INDEX_op_muluh_i64;

            have_opc_new2 = TCG_TARGET_HAS_muluh_i64;

            goto do_mul2;

        case INDEX_op_muls2_i64:

            opc_new = INDEX_op_mul_i64;

            opc_new2 = INDEX_op_mulsh_i64;

            have_opc_new2 = TCG_TARGET_HAS_mulsh_i64;

            goto do_mul2;

        do_mul2:

            nb_iargs = 2;

            nb_oargs = 2;

            if (arg_temp(op->args[1])->state == TS_DEAD) {

                if (arg_temp(op->args[0])->state == TS_DEAD) {

                    /* Both parts of the operation are dead.  */

                    goto do_remove;

                }

                /* The high part of the operation is dead; generate the low. */

                op->opc = opc = opc_new;

                op->args[1] = op->args[2];

                op->args[2] = op->args[3];

            } else if (arg_temp(op->args[0])->state == TS_DEAD && have_opc_new2) {

                /* The low part of the operation is dead; generate the high. */

                op->opc = opc = opc_new2;

                op->args[0] = op->args[1];

                op->args[1] = op->args[2];

                op->args[2] = op->args[3];

            } else {

                goto do_not_remove;

            }

            /* Mark the single-word operation live.  */

            nb_oargs = 1;

            goto do_not_remove;



        default:

            /* XXX: optimize by hardcoding common cases (e.g. triadic ops) */

            nb_iargs = def->nb_iargs;

            nb_oargs = def->nb_oargs;



            /* Test if the operation can be removed because all

               its outputs are dead. We assume that nb_oargs == 0

               implies side effects */

            if (!(def->flags & TCG_OPF_SIDE_EFFECTS) && nb_oargs != 0) {

                for (i = 0; i < nb_oargs; i++) {

                    if (arg_temp(op->args[i])->state != TS_DEAD) {

                        goto do_not_remove;

                    }

                }

            do_remove:

                tcg_op_remove(s, op);

            } else {

            do_not_remove:

                /* output args are dead */

                for (i = 0; i < nb_oargs; i++) {

                    arg_ts = arg_temp(op->args[i]);

                    if (arg_ts->state & TS_DEAD) {

                        arg_life |= DEAD_ARG << i;

                    }

                    if (arg_ts->state & TS_MEM) {

                        arg_life |= SYNC_ARG << i;

                    }

                    arg_ts->state = TS_DEAD;

                }



                /* if end of basic block, update */

                if (def->flags & TCG_OPF_BB_END) {

                    tcg_la_bb_end(s);

                } else if (def->flags & TCG_OPF_SIDE_EFFECTS) {

                    /* globals should be synced to memory */

                    for (i = 0; i < nb_globals; i++) {

                        s->temps[i].state |= TS_MEM;

                    }

                }



                /* record arguments that die in this opcode */

                for (i = nb_oargs; i < nb_oargs + nb_iargs; i++) {

                    arg_ts = arg_temp(op->args[i]);

                    if (arg_ts->state & TS_DEAD) {

                        arg_life |= DEAD_ARG << i;

                    }

                }

                /* input arguments are live for preceding opcodes */

                for (i = nb_oargs; i < nb_oargs + nb_iargs; i++) {

                    arg_temp(op->args[i])->state &= ~TS_DEAD;

                }

            }

            break;

        }

        op->life = arg_life;

    }

}
