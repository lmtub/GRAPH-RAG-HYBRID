int tcg_gen_code(TCGContext *s, TranslationBlock *tb)

{

    int i, oi, oi_next, num_insns;



#ifdef CONFIG_PROFILER

    {

        int n;



        n = s->gen_last_op_idx + 1;

        s->op_count += n;

        if (n > s->op_count_max) {

            s->op_count_max = n;

        }



        n = s->nb_temps;

        s->temp_count += n;

        if (n > s->temp_count_max) {

            s->temp_count_max = n;

        }

    }

#endif



#ifdef DEBUG_DISAS

    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP)

                 && qemu_log_in_addr_range(tb->pc))) {

        qemu_log("OP:\n");

        tcg_dump_ops(s);

        qemu_log("\n");

    }

#endif



#ifdef CONFIG_PROFILER

    s->opt_time -= profile_getclock();

#endif



#ifdef USE_TCG_OPTIMIZATIONS

    tcg_optimize(s);

#endif



#ifdef CONFIG_PROFILER

    s->opt_time += profile_getclock();

    s->la_time -= profile_getclock();

#endif



    tcg_liveness_analysis(s);



#ifdef CONFIG_PROFILER

    s->la_time += profile_getclock();

#endif



#ifdef DEBUG_DISAS

    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP_OPT)

                 && qemu_log_in_addr_range(tb->pc))) {

        qemu_log("OP after optimization and liveness analysis:\n");

        tcg_dump_ops(s);

        qemu_log("\n");

    }

#endif



    tcg_reg_alloc_start(s);



    s->code_buf = tb->tc_ptr;

    s->code_ptr = tb->tc_ptr;



    tcg_out_tb_init(s);



    num_insns = -1;

    for (oi = s->gen_first_op_idx; oi >= 0; oi = oi_next) {

        TCGOp * const op = &s->gen_op_buf[oi];

        TCGArg * const args = &s->gen_opparam_buf[op->args];

        TCGOpcode opc = op->opc;

        const TCGOpDef *def = &tcg_op_defs[opc];

        uint16_t dead_args = s->op_dead_args[oi];

        uint8_t sync_args = s->op_sync_args[oi];



        oi_next = op->next;

#ifdef CONFIG_PROFILER

        tcg_table_op_count[opc]++;

#endif



        switch (opc) {

        case INDEX_op_mov_i32:

        case INDEX_op_mov_i64:

            tcg_reg_alloc_mov(s, def, args, dead_args, sync_args);

            break;

        case INDEX_op_movi_i32:

        case INDEX_op_movi_i64:

            tcg_reg_alloc_movi(s, args, dead_args, sync_args);

            break;

        case INDEX_op_insn_start:

            if (num_insns >= 0) {

                s->gen_insn_end_off[num_insns] = tcg_current_code_size(s);

            }

            num_insns++;

            for (i = 0; i < TARGET_INSN_START_WORDS; ++i) {

                target_ulong a;

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

                a = ((target_ulong)args[i * 2 + 1] << 32) | args[i * 2];

#else

                a = args[i];

#endif

                s->gen_insn_data[num_insns][i] = a;

            }

            break;

        case INDEX_op_discard:

            temp_dead(s, &s->temps[args[0]]);

            break;

        case INDEX_op_set_label:

            tcg_reg_alloc_bb_end(s, s->reserved_regs);

            tcg_out_label(s, arg_label(args[0]), s->code_ptr);

            break;

        case INDEX_op_call:

            tcg_reg_alloc_call(s, op->callo, op->calli, args,

                               dead_args, sync_args);

            break;

        default:

            /* Sanity check that we've not introduced any unhandled opcodes. */

            if (def->flags & TCG_OPF_NOT_PRESENT) {

                tcg_abort();

            }

            /* Note: in order to speed up the code, it would be much

               faster to have specialized register allocator functions for

               some common argument patterns */

            tcg_reg_alloc_op(s, def, opc, args, dead_args, sync_args);

            break;

        }

#ifndef NDEBUG

        check_regs(s);

#endif

        /* Test for (pending) buffer overflow.  The assumption is that any

           one operation beginning below the high water mark cannot overrun

           the buffer completely.  Thus we can test for overflow after

           generating code without having to check during generation.  */

        if (unlikely((void *)s->code_ptr > s->code_gen_highwater)) {

            return -1;

        }

    }

    tcg_debug_assert(num_insns >= 0);

    s->gen_insn_end_off[num_insns] = tcg_current_code_size(s);



    /* Generate TB finalization at the end of block */

    if (!tcg_out_tb_finalize(s)) {

        return -1;

    }



    /* flush instruction cache */

    flush_icache_range((uintptr_t)s->code_buf, (uintptr_t)s->code_ptr);



    return tcg_current_code_size(s);

}
