static void tcg_reg_alloc_call(TCGContext *s, int nb_oargs, int nb_iargs,

                               const TCGArg * const args, uint16_t dead_args,

                               uint8_t sync_args)

{

    int flags, nb_regs, i;

    TCGReg reg;

    TCGArg arg;

    TCGTemp *ts;

    intptr_t stack_offset;

    size_t call_stack_size;

    tcg_insn_unit *func_addr;

    int allocate_args;

    TCGRegSet allocated_regs;



    func_addr = (tcg_insn_unit *)(intptr_t)args[nb_oargs + nb_iargs];

    flags = args[nb_oargs + nb_iargs + 1];



    nb_regs = ARRAY_SIZE(tcg_target_call_iarg_regs);

    if (nb_regs > nb_iargs) {

        nb_regs = nb_iargs;

    }



    /* assign stack slots first */

    call_stack_size = (nb_iargs - nb_regs) * sizeof(tcg_target_long);

    call_stack_size = (call_stack_size + TCG_TARGET_STACK_ALIGN - 1) & 

        ~(TCG_TARGET_STACK_ALIGN - 1);

    allocate_args = (call_stack_size > TCG_STATIC_CALL_ARGS_SIZE);

    if (allocate_args) {

        /* XXX: if more than TCG_STATIC_CALL_ARGS_SIZE is needed,

           preallocate call stack */

        tcg_abort();

    }



    stack_offset = TCG_TARGET_CALL_STACK_OFFSET;

    for(i = nb_regs; i < nb_iargs; i++) {

        arg = args[nb_oargs + i];

#ifdef TCG_TARGET_STACK_GROWSUP

        stack_offset -= sizeof(tcg_target_long);

#endif

        if (arg != TCG_CALL_DUMMY_ARG) {

            ts = &s->temps[arg];

            temp_load(s, ts, tcg_target_available_regs[ts->type],

                      s->reserved_regs);

            tcg_out_st(s, ts->type, ts->reg, TCG_REG_CALL_STACK, stack_offset);

        }

#ifndef TCG_TARGET_STACK_GROWSUP

        stack_offset += sizeof(tcg_target_long);

#endif

    }

    

    /* assign input registers */

    tcg_regset_set(allocated_regs, s->reserved_regs);

    for(i = 0; i < nb_regs; i++) {

        arg = args[nb_oargs + i];

        if (arg != TCG_CALL_DUMMY_ARG) {

            ts = &s->temps[arg];

            reg = tcg_target_call_iarg_regs[i];

            tcg_reg_free(s, reg, allocated_regs);



            if (ts->val_type == TEMP_VAL_REG) {

                if (ts->reg != reg) {

                    tcg_out_mov(s, ts->type, reg, ts->reg);

                }

            } else {

                TCGRegSet arg_set;



                tcg_regset_clear(arg_set);

                tcg_regset_set_reg(arg_set, reg);

                temp_load(s, ts, arg_set, allocated_regs);

            }



            tcg_regset_set_reg(allocated_regs, reg);

        }

    }

    

    /* mark dead temporaries and free the associated registers */

    for(i = nb_oargs; i < nb_iargs + nb_oargs; i++) {

        if (IS_DEAD_ARG(i)) {

            temp_dead(s, &s->temps[args[i]]);

        }

    }

    

    /* clobber call registers */

    for (i = 0; i < TCG_TARGET_NB_REGS; i++) {

        if (tcg_regset_test_reg(tcg_target_call_clobber_regs, i)) {

            tcg_reg_free(s, i, allocated_regs);

        }

    }



    /* Save globals if they might be written by the helper, sync them if

       they might be read. */

    if (flags & TCG_CALL_NO_READ_GLOBALS) {

        /* Nothing to do */

    } else if (flags & TCG_CALL_NO_WRITE_GLOBALS) {

        sync_globals(s, allocated_regs);

    } else {

        save_globals(s, allocated_regs);

    }



    tcg_out_call(s, func_addr);



    /* assign output registers and emit moves if needed */

    for(i = 0; i < nb_oargs; i++) {

        arg = args[i];

        ts = &s->temps[arg];

        reg = tcg_target_call_oarg_regs[i];

        assert(s->reg_to_temp[reg] == NULL);



        if (ts->fixed_reg) {

            if (ts->reg != reg) {

                tcg_out_mov(s, ts->type, ts->reg, reg);

            }

        } else {

            if (ts->val_type == TEMP_VAL_REG) {

                s->reg_to_temp[ts->reg] = NULL;

            }

            ts->val_type = TEMP_VAL_REG;

            ts->reg = reg;

            ts->mem_coherent = 0;

            s->reg_to_temp[reg] = ts;

            if (NEED_SYNC_ARG(i)) {

                tcg_reg_sync(s, reg, allocated_regs);

            }

            if (IS_DEAD_ARG(i)) {

                temp_dead(s, ts);

            }

        }

    }

}
