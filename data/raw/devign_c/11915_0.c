gen_intermediate_code_internal(CPUState * env, TranslationBlock * tb,

                               int search_pc)

{

    DisasContext ctx;

    target_ulong pc_start;

    static uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int i, ii;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.flags = (uint32_t)tb->flags;

    ctx.bstate = BS_NONE;

    ctx.sr = env->sr;

    ctx.fpscr = env->fpscr;

    ctx.memidx = (env->sr & SR_MD) ? 1 : 0;

    /* We don't know if the delayed pc came from a dynamic or static branch,

       so assume it is a dynamic branch.  */

    ctx.delayed_pc = -1; /* use delayed pc from env pointer */

    ctx.tb = tb;

    ctx.singlestep_enabled = env->singlestep_enabled;

    ctx.features = env->features;

    ctx.has_movcal = (tb->flags & TB_FLAG_PENDING_MOVCA);



#ifdef DEBUG_DISAS

    qemu_log_mask(CPU_LOG_TB_CPU,

                 "------------------------------------------------\n");

    log_cpu_state_mask(CPU_LOG_TB_CPU, env, 0);

#endif



    ii = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;

    gen_icount_start();

    while (ctx.bstate == BS_NONE && gen_opc_ptr < gen_opc_end) {

        if (unlikely(!TAILQ_EMPTY(&env->breakpoints))) {

            TAILQ_FOREACH(bp, &env->breakpoints, entry) {

                if (ctx.pc == bp->pc) {

		    /* We have hit a breakpoint - make sure PC is up-to-date */

		    tcg_gen_movi_i32(cpu_pc, ctx.pc);

		    gen_helper_debug();

		    ctx.bstate = BS_EXCP;

		    break;

		}

	    }

	}

        if (search_pc) {

            i = gen_opc_ptr - gen_opc_buf;

            if (ii < i) {

                ii++;

                while (ii < i)

                    gen_opc_instr_start[ii++] = 0;

            }

            gen_opc_pc[ii] = ctx.pc;

            gen_opc_hflags[ii] = ctx.flags;

            gen_opc_instr_start[ii] = 1;

            gen_opc_icount[ii] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

#if 0

	fprintf(stderr, "Loading opcode at address 0x%08x\n", ctx.pc);

	fflush(stderr);

#endif

	ctx.opcode = lduw_code(ctx.pc);

	decode_opc(&ctx);

        num_insns++;

	ctx.pc += 2;

	if ((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0)

	    break;

	if (env->singlestep_enabled)

	    break;

        if (num_insns >= max_insns)

            break;

        if (singlestep)

            break;

    }

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (env->singlestep_enabled) {

        tcg_gen_movi_i32(cpu_pc, ctx.pc);

        gen_helper_debug();

    } else {

	switch (ctx.bstate) {

        case BS_STOP:

            /* gen_op_interrupt_restart(); */

            /* fall through */

        case BS_NONE:

            if (ctx.flags) {

                gen_store_flags(ctx.flags | DELAY_SLOT_CLEARME);

	    }

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_EXCP:

            /* gen_op_interrupt_restart(); */

            tcg_gen_exit_tb(0);

            break;

        case BS_BRANCH:

        default:

            break;

	}

    }



    gen_icount_end(tb, num_insns);

    *gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        i = gen_opc_ptr - gen_opc_buf;

        ii++;

        while (ii <= i)

            gen_opc_instr_start[ii++] = 0;

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

#ifdef SH4_DEBUG_DISAS

    qemu_log_mask(CPU_LOG_TB_IN_ASM, "\n");

#endif

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

	qemu_log("IN:\n");	/* , lookup_symbol(pc_start)); */

	log_target_disas(pc_start, ctx.pc - pc_start, 0);

	qemu_log("\n");

    }

#endif

}
