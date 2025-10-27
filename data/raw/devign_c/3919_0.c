static inline int tcg_gen_code_common(TCGContext *s,

                                      tcg_insn_unit *gen_code_buf,

                                      long search_pc)

{

    int oi, oi_next;



#ifdef DEBUG_DISAS

    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP))) {

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

    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP_OPT))) {

        qemu_log("OP after optimization and liveness analysis:\n");

        tcg_dump_ops(s);

        qemu_log("\n");

    }

#endif



    tcg_reg_alloc_start(s);



    s->code_buf = gen_code_buf;

    s->code_ptr = gen_code_buf;



    tcg_out_tb_init(s);



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

        case INDEX_op_debug_insn_start:

            break;

        case INDEX_op_discard:

            temp_dead(s, args[0]);

            break;

        case INDEX_op_set_label:

            tcg_reg_alloc_bb_end(s, s->reserved_regs);

            tcg_out_label(s, args[0], s->code_ptr);

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

        if (search_pc >= 0 && search_pc < tcg_current_code_size(s)) {

            return oi;

        }

#ifndef NDEBUG

        check_regs(s);

#endif

    }



    /* Generate TB finalization at the end of block */

    tcg_out_tb_finalize(s);

    return -1;

}
