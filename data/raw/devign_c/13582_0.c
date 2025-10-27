static inline void gen_intermediate_code_internal(UniCore32CPU *cpu,

        TranslationBlock *tb, bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUUniCore32State *env = &cpu->env;

    DisasContext dc1, *dc = &dc1;

    CPUBreakpoint *bp;

    uint16_t *gen_opc_end;

    int j, lj;

    target_ulong pc_start;

    uint32_t next_page_start;

    int num_insns;

    int max_insns;



    /* generate intermediate code */

    num_temps = 0;



    pc_start = tb->pc;



    dc->tb = tb;



    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    dc->is_jmp = DISAS_NEXT;

    dc->pc = pc_start;

    dc->singlestep_enabled = cs->singlestep_enabled;

    dc->condjmp = 0;

    cpu_F0s = tcg_temp_new_i32();

    cpu_F1s = tcg_temp_new_i32();

    cpu_F0d = tcg_temp_new_i64();

    cpu_F1d = tcg_temp_new_i64();

    next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

    lj = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



#ifndef CONFIG_USER_ONLY

    if ((env->uncached_asr & ASR_M) == ASR_MODE_USER) {

        dc->user = 1;

    } else {

        dc->user = 0;

    }

#endif



    gen_tb_start();

    do {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == dc->pc) {

                    gen_set_pc_im(dc->pc);

                    gen_exception(EXCP_DEBUG);

                    dc->is_jmp = DISAS_JUMP;

                    /* Advance PC so that clearing the breakpoint will

                       invalidate this TB.  */

                    dc->pc += 2; /* FIXME */

                    goto done_generating;

                }

            }

        }

        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j) {

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

                }

            }

            tcg_ctx.gen_opc_pc[lj] = dc->pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }



        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        disas_uc32_insn(env, dc);



        if (num_temps) {

            fprintf(stderr, "Internal resource leak before %08x\n", dc->pc);

            num_temps = 0;

        }



        if (dc->condjmp && !dc->is_jmp) {

            gen_set_label(dc->condlabel);

            dc->condjmp = 0;

        }

        /* Translation stops when a conditional branch is encountered.

         * Otherwise the subsequent code could get translated several times.

         * Also stop translation when a page boundary is reached.  This

         * ensures prefetch aborts occur at the right place.  */

        num_insns++;

    } while (!dc->is_jmp && tcg_ctx.gen_opc_ptr < gen_opc_end &&

             !cs->singlestep_enabled &&

             !singlestep &&

             dc->pc < next_page_start &&

             num_insns < max_insns);



    if (tb->cflags & CF_LAST_IO) {

        if (dc->condjmp) {

            /* FIXME:  This can theoretically happen with self-modifying

               code.  */

            cpu_abort(cs, "IO on conditional branch instruction");

        }

        gen_io_end();

    }



    /* At this stage dc->condjmp will only be set when the skipped

       instruction was a conditional branch or trap, and the PC has

       already been written.  */

    if (unlikely(cs->singlestep_enabled)) {

        /* Make sure the pc is updated, and raise a debug exception.  */

        if (dc->condjmp) {

            if (dc->is_jmp == DISAS_SYSCALL) {

                gen_exception(UC32_EXCP_PRIV);

            } else {

                gen_exception(EXCP_DEBUG);

            }

            gen_set_label(dc->condlabel);

        }

        if (dc->condjmp || !dc->is_jmp) {

            gen_set_pc_im(dc->pc);

            dc->condjmp = 0;

        }

        if (dc->is_jmp == DISAS_SYSCALL && !dc->condjmp) {

            gen_exception(UC32_EXCP_PRIV);

        } else {

            gen_exception(EXCP_DEBUG);

        }

    } else {

        /* While branches must always occur at the end of an IT block,

           there are a few other things that can cause us to terminate

           the TB in the middel of an IT block:

            - Exception generating instructions (bkpt, swi, undefined).

            - Page boundaries.

            - Hardware watchpoints.

           Hardware breakpoints have already been handled and skip this code.

         */

        switch (dc->is_jmp) {

        case DISAS_NEXT:

            gen_goto_tb(dc, 1, dc->pc);

            break;

        default:

        case DISAS_JUMP:

        case DISAS_UPDATE:

            /* indicate that the hash table must be used to find the next TB */

            tcg_gen_exit_tb(0);

            break;

        case DISAS_TB_JUMP:

            /* nothing more to generate */

            break;

        case DISAS_SYSCALL:

            gen_exception(UC32_EXCP_PRIV);

            break;

        }

        if (dc->condjmp) {

            gen_set_label(dc->condlabel);

            gen_goto_tb(dc, 1, dc->pc);

            dc->condjmp = 0;

        }

    }



done_generating:

    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("----------------\n");

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, dc->pc - pc_start, 0);

        qemu_log("\n");

    }

#endif

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j) {

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

        }

    } else {

        tb->size = dc->pc - pc_start;

        tb->icount = num_insns;

    }

}
