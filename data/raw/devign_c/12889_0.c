void gen_intermediate_code(CPUSH4State * env, struct TranslationBlock *tb)

{

    SuperHCPU *cpu = sh_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    DisasContext ctx;

    target_ulong pc_start;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    ctx.pc = pc_start;

    ctx.tbflags = (uint32_t)tb->flags;

    ctx.envflags = tb->flags & DELAY_SLOT_MASK;

    ctx.bstate = BS_NONE;

    ctx.memidx = (ctx.tbflags & (1u << SR_MD)) == 0 ? 1 : 0;

    /* We don't know if the delayed pc came from a dynamic or static branch,

       so assume it is a dynamic branch.  */

    ctx.delayed_pc = -1; /* use delayed pc from env pointer */

    ctx.tb = tb;

    ctx.singlestep_enabled = cs->singlestep_enabled;

    ctx.features = env->features;

    ctx.has_movcal = (ctx.tbflags & TB_FLAG_PENDING_MOVCA);



    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }

    if (max_insns > TCG_MAX_INSNS) {

        max_insns = TCG_MAX_INSNS;

    }



    gen_tb_start(tb);

    while (ctx.bstate == BS_NONE && !tcg_op_buf_full()) {

        tcg_gen_insn_start(ctx.pc, ctx.envflags);

        num_insns++;



        if (unlikely(cpu_breakpoint_test(cs, ctx.pc, BP_ANY))) {

            /* We have hit a breakpoint - make sure PC is up-to-date */

            gen_save_cpu_state(&ctx, true);

            gen_helper_debug(cpu_env);

            ctx.bstate = BS_EXCP;

            /* The address covered by the breakpoint must be included in

               [tb->pc, tb->pc + tb->size) in order to for it to be

               properly cleared -- thus we increment the PC here so that

               the logic setting tb->size below does the right thing.  */

            ctx.pc += 2;

            break;

        }



        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO)) {

            gen_io_start();

        }



        ctx.opcode = cpu_lduw_code(env, ctx.pc);

	decode_opc(&ctx);

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

        gen_save_cpu_state(&ctx, true);

        gen_helper_debug(cpu_env);

    } else {

	switch (ctx.bstate) {

        case BS_STOP:

            gen_save_cpu_state(&ctx, true);

            tcg_gen_exit_tb(0);

            break;

        case BS_NONE:

            gen_save_cpu_state(&ctx, false);

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_EXCP:

            /* fall through */

        case BS_BRANCH:

        default:

            break;

	}

    }



    gen_tb_end(tb, num_insns);



    tb->size = ctx.pc - pc_start;

    tb->icount = num_insns;



#ifdef DEBUG_DISAS

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)

        && qemu_log_in_addr_range(pc_start)) {

        qemu_log_lock();

	qemu_log("IN:\n");	/* , lookup_symbol(pc_start)); */

        log_target_disas(cs, pc_start, ctx.pc - pc_start, 0);

	qemu_log("\n");

        qemu_log_unlock();

    }

#endif

}
