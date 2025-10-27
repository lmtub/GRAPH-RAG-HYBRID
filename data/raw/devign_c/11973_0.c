static inline int tcg_gen_code_common(TCGContext *s, uint8_t *gen_code_buf,

                                      long search_pc)

{

    int opc, op_index, macro_op_index;

    const TCGOpDef *def;

    unsigned int dead_iargs;

    const TCGArg *args;



#ifdef DEBUG_DISAS

    if (unlikely(loglevel & CPU_LOG_TB_OP)) {

        fprintf(logfile, "OP:\n");

        tcg_dump_ops(s, logfile);

        fprintf(logfile, "\n");

    }

#endif



    tcg_liveness_analysis(s);



#ifdef DEBUG_DISAS

    if (unlikely(loglevel & CPU_LOG_TB_OP_OPT)) {

        fprintf(logfile, "OP after la:\n");

        tcg_dump_ops(s, logfile);

        fprintf(logfile, "\n");

    }

#endif



    tcg_reg_alloc_start(s);



    s->code_buf = gen_code_buf;

    s->code_ptr = gen_code_buf;



    macro_op_index = -1;

    args = gen_opparam_buf;

    op_index = 0;



    for(;;) {

        opc = gen_opc_buf[op_index];

#ifdef CONFIG_PROFILER

        dyngen_table_op_count[opc]++;

#endif

        def = &tcg_op_defs[opc];

#if 0

        printf("%s: %d %d %d\n", def->name,

               def->nb_oargs, def->nb_iargs, def->nb_cargs);

        //        dump_regs(s);

#endif

        switch(opc) {

        case INDEX_op_mov_i32:

#if TCG_TARGET_REG_BITS == 64

        case INDEX_op_mov_i64:

#endif

            dead_iargs = s->op_dead_iargs[op_index];

            tcg_reg_alloc_mov(s, def, args, dead_iargs);

            break;

        case INDEX_op_nop:

        case INDEX_op_nop1:

        case INDEX_op_nop2:

        case INDEX_op_nop3:

            break;

        case INDEX_op_nopn:

            args += args[0];

            goto next;

        case INDEX_op_discard:

            {

                TCGTemp *ts;

                ts = &s->temps[args[0]];

                /* mark the temporary as dead */

                if (ts->val_type != TEMP_VAL_CONST && !ts->fixed_reg) {

                    if (ts->val_type == TEMP_VAL_REG)

                        s->reg_to_temp[ts->reg] = -1;

                    ts->val_type = TEMP_VAL_DEAD;

                }

            }

            break;

        case INDEX_op_macro_goto:

            macro_op_index = op_index; /* only used for exceptions */

            op_index = args[0] - 1;

            args = gen_opparam_buf + args[1];

            goto next;

        case INDEX_op_macro_end:

            macro_op_index = -1; /* only used for exceptions */

            op_index = args[0] - 1;

            args = gen_opparam_buf + args[1];

            goto next;

        case INDEX_op_macro_start:

            /* must never happen here */

            tcg_abort();

        case INDEX_op_set_label:

            tcg_reg_alloc_bb_end(s);

            tcg_out_label(s, args[0], (long)s->code_ptr);

            break;

        case INDEX_op_call:

            dead_iargs = s->op_dead_iargs[op_index];

            args += tcg_reg_alloc_call(s, def, opc, args, dead_iargs);

            goto next;

        case INDEX_op_end:

            goto the_end;



#ifndef CONFIG_NO_DYNGEN_OP

        case 0 ... INDEX_op_end - 1:

            /* legacy dyngen ops */

#ifdef CONFIG_PROFILER

            {

                extern int64_t dyngen_old_op_count;

                dyngen_old_op_count++;

            }

#endif

            tcg_reg_alloc_bb_end(s);

            if (search_pc >= 0) {

                s->code_ptr += def->copy_size;

                args += def->nb_args;

            } else {

                args = dyngen_op(s, opc, args);

            }

            goto next;

#endif

        default:

            /* Note: in order to speed up the code, it would be much

               faster to have specialized register allocator functions for

               some common argument patterns */

            dead_iargs = s->op_dead_iargs[op_index];

            tcg_reg_alloc_op(s, def, opc, args, dead_iargs);

            break;

        }

        args += def->nb_args;

    next: ;

        if (search_pc >= 0 && search_pc < s->code_ptr - gen_code_buf) {

            if (macro_op_index >= 0)

                return macro_op_index;

            else

                return op_index;

        }

        op_index++;

#ifndef NDEBUG

        check_regs(s);

#endif

    }

 the_end:

    return -1;

}
