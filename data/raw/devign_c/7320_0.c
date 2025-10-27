static inline void gen_intermediate_code_internal(AlphaCPU *cpu,

                                                  TranslationBlock *tb,

                                                  bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUAlphaState *env = &cpu->env;

    DisasContext ctx, *ctxp = &ctx;

    target_ulong pc_start;

    target_ulong pc_mask;

    uint32_t insn;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj = -1;

    ExitStatus ret;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;



    ctx.tb = tb;

    ctx.pc = pc_start;

    ctx.mem_idx = cpu_mmu_index(env);

    ctx.implver = env->implver;

    ctx.singlestep_enabled = cs->singlestep_enabled;



    /* ??? Every TB begins with unset rounding mode, to be initialized on

       the first fp insn of the TB.  Alternately we could define a proper

       default for every TB (e.g. QUAL_RM_N or QUAL_RM_D) and make sure

       to reset the FP_STATUS to that default at the end of any TB that

       changes the default.  We could even (gasp) dynamiclly figure out

       what default would be most efficient given the running program.  */

    ctx.tb_rm = -1;

    /* Similarly for flush-to-zero.  */

    ctx.tb_ftz = -1;



    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }



    if (in_superpage(&ctx, pc_start)) {

        pc_mask = (1ULL << 41) - 1;

    } else {

        pc_mask = ~TARGET_PAGE_MASK;

    }



    gen_tb_start();

    do {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (bp->pc == ctx.pc) {

                    gen_excp(&ctx, EXCP_DEBUG, 0);

                    break;

                }

            }

        }

        if (search_pc) {

            j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    tcg_ctx.gen_opc_instr_start[lj++] = 0;

            }

            tcg_ctx.gen_opc_pc[lj] = ctx.pc;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }

        insn = cpu_ldl_code(env, ctx.pc);

        num_insns++;



	if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

            tcg_gen_debug_insn_start(ctx.pc);

        }



        TCGV_UNUSED_I64(ctx.zero);

        TCGV_UNUSED_I64(ctx.sink);

        TCGV_UNUSED_I64(ctx.lit);



        ctx.pc += 4;

        ret = translate_one(ctxp, insn);



        if (!TCGV_IS_UNUSED_I64(ctx.sink)) {

            tcg_gen_discard_i64(ctx.sink);

            tcg_temp_free(ctx.sink);

        }

        if (!TCGV_IS_UNUSED_I64(ctx.zero)) {

            tcg_temp_free(ctx.zero);

        }

        if (!TCGV_IS_UNUSED_I64(ctx.lit)) {

            tcg_temp_free(ctx.lit);

        }



        /* If we reach a page boundary, are single stepping,

           or exhaust instruction count, stop generation.  */

        if (ret == NO_EXIT

            && ((ctx.pc & pc_mask) == 0

                || tcg_ctx.gen_opc_ptr >= gen_opc_end

                || num_insns >= max_insns

                || singlestep

                || ctx.singlestep_enabled)) {

            ret = EXIT_PC_STALE;

        }

    } while (ret == NO_EXIT);



    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }



    switch (ret) {

    case EXIT_GOTO_TB:

    case EXIT_NORETURN:

        break;

    case EXIT_PC_STALE:

        tcg_gen_movi_i64(cpu_pc, ctx.pc);

        /* FALLTHRU */

    case EXIT_PC_UPDATED:

        if (ctx.singlestep_enabled) {

            gen_excp_1(EXCP_DEBUG, 0);

        } else {

            tcg_gen_exit_tb(0);

        }

        break;

    default:

        abort();

    }



    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        lj++;

        while (lj <= j)

            tcg_ctx.gen_opc_instr_start[lj++] = 0;

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, ctx.pc - pc_start, 1);

        qemu_log("\n");

    }

#endif

}
