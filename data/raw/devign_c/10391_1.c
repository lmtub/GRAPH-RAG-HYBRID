static bool liveness_pass_2(TCGContext *s)

{

    int nb_globals = s->nb_globals;

    int nb_temps, i, oi, oi_next;

    bool changes = false;



    /* Create a temporary for each indirect global.  */

    for (i = 0; i < nb_globals; ++i) {

        TCGTemp *its = &s->temps[i];

        if (its->indirect_reg) {

            TCGTemp *dts = tcg_temp_alloc(s);

            dts->type = its->type;

            dts->base_type = its->base_type;

            its->state_ptr = dts;

        } else {

            its->state_ptr = NULL;

        }

        /* All globals begin dead.  */

        its->state = TS_DEAD;

    }

    for (nb_temps = s->nb_temps; i < nb_temps; ++i) {

        TCGTemp *its = &s->temps[i];

        its->state_ptr = NULL;

        its->state = TS_DEAD;

    }



    for (oi = s->gen_op_buf[0].next; oi != 0; oi = oi_next) {

        TCGOp *op = &s->gen_op_buf[oi];

        TCGOpcode opc = op->opc;

        const TCGOpDef *def = &tcg_op_defs[opc];

        TCGLifeData arg_life = op->life;

        int nb_iargs, nb_oargs, call_flags;

        TCGTemp *arg_ts, *dir_ts;



        oi_next = op->next;



        if (opc == INDEX_op_call) {

            nb_oargs = op->callo;

            nb_iargs = op->calli;

            call_flags = op->args[nb_oargs + nb_iargs + 1];

        } else {

            nb_iargs = def->nb_iargs;

            nb_oargs = def->nb_oargs;



            /* Set flags similar to how calls require.  */

            if (def->flags & TCG_OPF_BB_END) {

                /* Like writing globals: save_globals */

                call_flags = 0;

            } else if (def->flags & TCG_OPF_SIDE_EFFECTS) {

                /* Like reading globals: sync_globals */

                call_flags = TCG_CALL_NO_WRITE_GLOBALS;

            } else {

                /* No effect on globals.  */

                call_flags = (TCG_CALL_NO_READ_GLOBALS |

                              TCG_CALL_NO_WRITE_GLOBALS);

            }

        }



        /* Make sure that input arguments are available.  */

        for (i = nb_oargs; i < nb_iargs + nb_oargs; i++) {

            arg_ts = arg_temp(op->args[i]);

            if (arg_ts) {

                dir_ts = arg_ts->state_ptr;

                if (dir_ts && arg_ts->state == TS_DEAD) {

                    TCGOpcode lopc = (arg_ts->type == TCG_TYPE_I32

                                      ? INDEX_op_ld_i32

                                      : INDEX_op_ld_i64);

                    TCGOp *lop = tcg_op_insert_before(s, op, lopc, 3);



                    lop->args[0] = temp_arg(dir_ts);

                    lop->args[1] = temp_arg(arg_ts->mem_base);

                    lop->args[2] = arg_ts->mem_offset;



                    /* Loaded, but synced with memory.  */

                    arg_ts->state = TS_MEM;

                }

            }

        }



        /* Perform input replacement, and mark inputs that became dead.

           No action is required except keeping temp_state up to date

           so that we reload when needed.  */

        for (i = nb_oargs; i < nb_iargs + nb_oargs; i++) {

            arg_ts = arg_temp(op->args[i]);

            if (arg_ts) {

                dir_ts = arg_ts->state_ptr;

                if (dir_ts) {

                    op->args[i] = temp_arg(dir_ts);

                    changes = true;

                    if (IS_DEAD_ARG(i)) {

                        arg_ts->state = TS_DEAD;

                    }

                }

            }

        }



        /* Liveness analysis should ensure that the following are

           all correct, for call sites and basic block end points.  */

        if (call_flags & TCG_CALL_NO_READ_GLOBALS) {

            /* Nothing to do */

        } else if (call_flags & TCG_CALL_NO_WRITE_GLOBALS) {

            for (i = 0; i < nb_globals; ++i) {

                /* Liveness should see that globals are synced back,

                   that is, either TS_DEAD or TS_MEM.  */

                arg_ts = &s->temps[i];

                tcg_debug_assert(arg_ts->state_ptr == 0

                                 || arg_ts->state != 0);

            }

        } else {

            for (i = 0; i < nb_globals; ++i) {

                /* Liveness should see that globals are saved back,

                   that is, TS_DEAD, waiting to be reloaded.  */

                arg_ts = &s->temps[i];

                tcg_debug_assert(arg_ts->state_ptr == 0

                                 || arg_ts->state == TS_DEAD);

            }

        }



        /* Outputs become available.  */

        for (i = 0; i < nb_oargs; i++) {

            arg_ts = arg_temp(op->args[i]);

            dir_ts = arg_ts->state_ptr;

            if (!dir_ts) {

                continue;

            }

            op->args[i] = temp_arg(dir_ts);

            changes = true;



            /* The output is now live and modified.  */

            arg_ts->state = 0;



            /* Sync outputs upon their last write.  */

            if (NEED_SYNC_ARG(i)) {

                TCGOpcode sopc = (arg_ts->type == TCG_TYPE_I32

                                  ? INDEX_op_st_i32

                                  : INDEX_op_st_i64);

                TCGOp *sop = tcg_op_insert_after(s, op, sopc, 3);



                sop->args[0] = temp_arg(dir_ts);

                sop->args[1] = temp_arg(arg_ts->mem_base);

                sop->args[2] = arg_ts->mem_offset;



                arg_ts->state = TS_MEM;

            }

            /* Drop outputs that are dead.  */

            if (IS_DEAD_ARG(i)) {

                arg_ts->state = TS_DEAD;

            }

        }

    }



    return changes;

}
