gen_intermediate_code_internal(MIPSCPU *cpu, TranslationBlock *tb,

                               bool search_pc)

{

    CPUState *cs = CPU(cpu);

    CPUMIPSState *env = &cpu->env;

    DisasContext ctx;

    target_ulong pc_start;

    uint16_t *gen_opc_end;

    CPUBreakpoint *bp;

    int j, lj = -1;

    int num_insns;

    int max_insns;

    int insn_bytes;

    int is_branch;



    if (search_pc)

        qemu_log("search pc %d\n", search_pc);



    pc_start = tb->pc;

    gen_opc_end = tcg_ctx.gen_opc_buf + OPC_MAX_SIZE;

    ctx.pc = pc_start;

    ctx.saved_pc = -1;

    ctx.singlestep_enabled = cs->singlestep_enabled;

    ctx.insn_flags = env->insn_flags;

    ctx.tb = tb;

    ctx.bstate = BS_NONE;

    /* Restore delay slot state from the tb context.  */

    ctx.hflags = (uint32_t)tb->flags; /* FIXME: maybe use 64 bits here? */

    restore_cpu_state(env, &ctx);

#ifdef CONFIG_USER_ONLY

        ctx.mem_idx = MIPS_HFLAG_UM;

#else

        ctx.mem_idx = ctx.hflags & MIPS_HFLAG_KSU;

#endif

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0)

        max_insns = CF_COUNT_MASK;

    LOG_DISAS("\ntb %p idx %d hflags %04x\n", tb, ctx.mem_idx, ctx.hflags);

    gen_tb_start();

    while (ctx.bstate == BS_NONE) {

        if (unlikely(!QTAILQ_EMPTY(&env->breakpoints))) {

            QTAILQ_FOREACH(bp, &env->breakpoints, entry) {

                if (bp->pc == ctx.pc) {

                    save_cpu_state(&ctx, 1);

                    ctx.bstate = BS_BRANCH;

                    gen_helper_0e0i(raise_exception, EXCP_DEBUG);

                    /* Include the breakpoint location or the tb won't

                     * be flushed when it must be.  */

                    ctx.pc += 4;

                    goto done_generating;

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

            gen_opc_hflags[lj] = ctx.hflags & MIPS_HFLAG_BMASK;

            gen_opc_btarget[lj] = ctx.btarget;

            tcg_ctx.gen_opc_instr_start[lj] = 1;

            tcg_ctx.gen_opc_icount[lj] = num_insns;

        }

        if (num_insns + 1 == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();



        is_branch = 0;

        if (!(ctx.hflags & MIPS_HFLAG_M16)) {

            ctx.opcode = cpu_ldl_code(env, ctx.pc);

            insn_bytes = 4;

            decode_opc(env, &ctx, &is_branch);

        } else if (ctx.insn_flags & ASE_MICROMIPS) {

            ctx.opcode = cpu_lduw_code(env, ctx.pc);

            insn_bytes = decode_micromips_opc(env, &ctx, &is_branch);

        } else if (ctx.insn_flags & ASE_MIPS16) {

            ctx.opcode = cpu_lduw_code(env, ctx.pc);

            insn_bytes = decode_mips16_opc(env, &ctx, &is_branch);

        } else {

            generate_exception(&ctx, EXCP_RI);

            ctx.bstate = BS_STOP;

            break;

        }

        if (!is_branch) {

            handle_delay_slot(&ctx, insn_bytes);

        }

        ctx.pc += insn_bytes;



        num_insns++;



        /* Execute a branch and its delay slot as a single instruction.

           This is what GDB expects and is consistent with what the

           hardware does (e.g. if a delay slot instruction faults, the

           reported PC is the PC of the branch).  */

        if (cs->singlestep_enabled && (ctx.hflags & MIPS_HFLAG_BMASK) == 0) {

            break;

        }



        if ((ctx.pc & (TARGET_PAGE_SIZE - 1)) == 0)

            break;



        if (tcg_ctx.gen_opc_ptr >= gen_opc_end) {

            break;

        }



        if (num_insns >= max_insns)

            break;



        if (singlestep)

            break;

    }

    if (tb->cflags & CF_LAST_IO) {

        gen_io_end();

    }

    if (cs->singlestep_enabled && ctx.bstate != BS_BRANCH) {

        save_cpu_state(&ctx, ctx.bstate == BS_NONE);

        gen_helper_0e0i(raise_exception, EXCP_DEBUG);

    } else {

        switch (ctx.bstate) {

        case BS_STOP:

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_NONE:

            save_cpu_state(&ctx, 0);

            gen_goto_tb(&ctx, 0, ctx.pc);

            break;

        case BS_EXCP:

            tcg_gen_exit_tb(0);

            break;

        case BS_BRANCH:

        default:

            break;

        }

    }

done_generating:

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

    LOG_DISAS("\n");

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)) {

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(env, pc_start, ctx.pc - pc_start, 0);

        qemu_log("\n");

    }

#endif

}
