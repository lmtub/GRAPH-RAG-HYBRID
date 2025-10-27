gen_intermediate_code_internal(SuperHCPU *cpu, TranslationBlock *tb,

                               bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUSH4State *env = &cpu->env;

    DisasContext ctx;

    target_ulong pc_start;

    static uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int i, ii;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.flags = (uint32_t)tb->flags;

    ctx.bstate = BS_NONE;

    ctx.memidx = (ctx.flags & SR_MD) == 0 ? 1 : 0;

    /* We don't know if the delayed pc came from a dynamic or static branch,

       so assume it is a dynamic branch.  */

    ctx.delayed_pc = -1; /* use delayed pc from env pointer */

    ctx.tb = tb;

    ctx.singlestep_enabled = cs->singlestep_enabled;

    ctx.features = env->features;

    ctx.has_movcal = (ctx.flags & TB_FLAG_PENDING_MOVCA);



    ii = -1;

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;

    gen_tb_start();

    while (ctx.bstate == BS_NONE && tcg_ctx.gen_opc_ptr < gen_opc_end) {

        if (unlikely(!QTAILQ_EMPTY(&cs->breakpoints))) {

            QTAILQ_FOREACH(bp, &cs->breakpoints, entry) {

                if (ctx.pc == bp->pc) {

		    /* We have hit a breakpoint - make sure PC is up-to-date */

		    tcg_gen_movi_i32(cpu_pc, ctx.pc);

                    gen_helper_debug(cpu_env);

                    ctx.bstate = BS_BRANCH;

		    break;

		}

	    }

	}

        if (search_pc) {

            i = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

            if (ii < i) {

                ii++;

                while (ii < i)

                    tcg_ctx.gen_opc_instr_start[ii++] = 0;

            }

            tcg_ctx.gen_opc_pc[ii] = ctx.pc;

            gen_opc_hflags[ii] = ctx.flags;

            tcg_ctx.gen_opc_instr_start[ii] = 1;

            tcg_ctx.gen_opc_icount[ii] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

#if 0

	fprintf(stderr, "Loading opcode at address 0x%08x\n", ctx.pc);

	fflush(stderr);

#endif

        ctx.opcode = cpu_lduw_code(env, ctx.pc);

	decode_opc(&ctx);

        num_insns++;

	ctx.pc += 2;

	if ((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0)

	    break;

        if (cs->singlestep_enabled) {

	    break;

        }

        if (num_insns >= max_insns)

            break;

        if (singlestep)

            break;

    }

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (cs->singlestep_enabled) {

        tcg_gen_movi_i32(cpu_pc, ctx.pc);

        gen_helper_debug(cpu_env);

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



    gen_tb_end(tb, num_insns);

    *tcg_ctx.gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        i = tcg_ctx.gen_opc_ptr - tcg_ctx.gen_opc_buf;

        ii++;

        while (ii <= i)

            tcg_ctx.gen_opc_instr_start[ii++] = 0;

    } else {

        tb->size = ctx.pc - pc_start;

        tb->icount = num_insns;

    }



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

	qemu_log("IN:\n");	/* , lookup_symbol(pc_start)); */

        log_target_disas(env, pc_start, ctx.pc - pc_start, 0);

	qemu_log("\n");

    }

#endif

}
