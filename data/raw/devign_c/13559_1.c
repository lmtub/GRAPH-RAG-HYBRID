void gen_intermediate_code(CPUPPCState *env, struct TranslationBlock *tb)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    CPUState *cs = CPU(cpu);

    DisasContext ctx, *ctxp = &ctx;

    opc_handler_t **table, *handler;

    target_ulong pc_start;

    int num_insns;

    int max_insns;



    pc_start = tb->pc;

    ctx.nip = pc_start;

    ctx.tb = tb;

    ctx.exception = POWERPC_EXCP_NONE;

    ctx.spr_cb = env->spr_cb;

    ctx.pr = msr_pr;

    ctx.mem_idx = env->dmmu_idx;

    ctx.dr = msr_dr;

#if !defined(CONFIG_USER_ONLY)

    ctx.hv = msr_hv || !env->has_hv_mode;

#endif

    ctx.insns_flags = env->insns_flags;

    ctx.insns_flags2 = env->insns_flags2;

    ctx.access_type = -1;

    ctx.le_mode = !!(env->hflags & (1 << MSR_LE));

    ctx.default_tcg_memop_mask = ctx.le_mode ? MO_LE : MO_BE;

#if defined(TARGET_PPC64)

    ctx.sf_mode = msr_is_64bit(env, env->msr);

    ctx.has_cfar = !!(env->flags & POWERPC_FLAG_CFAR);

#endif

    if (env->mmu_model == POWERPC_MMU_32B ||

        env->mmu_model == POWERPC_MMU_601 ||

        (env->mmu_model & POWERPC_MMU_64B))

            ctx.lazy_tlb_flush = true;



    ctx.fpu_enabled = !!msr_fp;

    if ((env->flags & POWERPC_FLAG_SPE) && msr_spe)

        ctx.spe_enabled = !!msr_spe;

    else

        ctx.spe_enabled = false;

    if ((env->flags & POWERPC_FLAG_VRE) && msr_vr)

        ctx.altivec_enabled = !!msr_vr;

    else

        ctx.altivec_enabled = false;

    if ((env->flags & POWERPC_FLAG_VSX) && msr_vsx) {

        ctx.vsx_enabled = !!msr_vsx;

    } else {

        ctx.vsx_enabled = false;

    }

#if defined(TARGET_PPC64)

    if ((env->flags & POWERPC_FLAG_TM) && msr_tm) {

        ctx.tm_enabled = !!msr_tm;

    } else {

        ctx.tm_enabled = false;

    }

#endif

    if ((env->flags & POWERPC_FLAG_SE) && msr_se)

        ctx.singlestep_enabled = CPU_SINGLE_STEP;

    else

        ctx.singlestep_enabled = 0;

    if ((env->flags & POWERPC_FLAG_BE) && msr_be)

        ctx.singlestep_enabled |= CPU_BRANCH_STEP;

    if (unlikely(cs->singlestep_enabled)) {

        ctx.singlestep_enabled |= GDBSTUB_SINGLE_STEP;

    }

#if defined (DO_SINGLE_STEP) && 0

    /* Single step trace mode */

    msr_se = 1;

#endif

    num_insns = 0;

    max_insns = tb->cflags & CF_COUNT_MASK;

    if (max_insns == 0) {

        max_insns = CF_COUNT_MASK;

    }

    if (max_insns > TCG_MAX_INSNS) {

        max_insns = TCG_MAX_INSNS;

    }



    gen_tb_start(tb);

    tcg_clear_temp_count();

    /* Set env in case of segfault during code fetch */

    while (ctx.exception == POWERPC_EXCP_NONE && !tcg_op_buf_full()) {

        tcg_gen_insn_start(ctx.nip);

        num_insns++;



        if (unlikely(cpu_breakpoint_test(cs, ctx.nip, BP_ANY))) {

            gen_debug_exception(ctxp);

            /* The address covered by the breakpoint must be included in

               [tb->pc, tb->pc + tb->size) in order to for it to be

               properly cleared -- thus we increment the PC here so that

               the logic setting tb->size below does the right thing.  */

            ctx.nip += 4;

            break;

        }



        LOG_DISAS("----------------\n");

        LOG_DISAS("nip=" TARGET_FMT_lx " super=%d ir=%d\n",

                  ctx.nip, ctx.mem_idx, (int)msr_ir);

        if (num_insns == max_insns && (tb->cflags & CF_LAST_IO))

            gen_io_start();

        if (unlikely(need_byteswap(&ctx))) {

            ctx.opcode = bswap32(cpu_ldl_code(env, ctx.nip));

        } else {

            ctx.opcode = cpu_ldl_code(env, ctx.nip);

        }

        LOG_DISAS("translate opcode %08x (%02x %02x %02x) (%s)\n",

                    ctx.opcode, opc1(ctx.opcode), opc2(ctx.opcode),

                    opc3(ctx.opcode), ctx.le_mode ? "little" : "big");

        ctx.nip += 4;

        table = env->opcodes;

        handler = table[opc1(ctx.opcode)];

        if (is_indirect_opcode(handler)) {

            table = ind_table(handler);

            handler = table[opc2(ctx.opcode)];

            if (is_indirect_opcode(handler)) {

                table = ind_table(handler);

                handler = table[opc3(ctx.opcode)];

            }

        }

        /* Is opcode *REALLY* valid ? */

        if (unlikely(handler->handler == &gen_invalid)) {

            qemu_log_mask(LOG_GUEST_ERROR, "invalid/unsupported opcode: "

                          "%02x - %02x - %02x (%08x) " TARGET_FMT_lx " %d\n",

                          opc1(ctx.opcode), opc2(ctx.opcode),

                          opc3(ctx.opcode), ctx.opcode, ctx.nip - 4, (int)msr_ir);

        } else {

            uint32_t inval;



            if (unlikely(handler->type & (PPC_SPE | PPC_SPE_SINGLE | PPC_SPE_DOUBLE) && Rc(ctx.opcode))) {

                inval = handler->inval2;

            } else {

                inval = handler->inval1;

            }



            if (unlikely((ctx.opcode & inval) != 0)) {

                qemu_log_mask(LOG_GUEST_ERROR, "invalid bits: %08x for opcode: "

                              "%02x - %02x - %02x (%08x) " TARGET_FMT_lx "\n",

                              ctx.opcode & inval, opc1(ctx.opcode),

                              opc2(ctx.opcode), opc3(ctx.opcode),

                              ctx.opcode, ctx.nip - 4);

                gen_inval_exception(ctxp, POWERPC_EXCP_INVAL_INVAL);

                break;

            }

        }

        (*(handler->handler))(&ctx);

#if defined(DO_PPC_STATISTICS)

        handler->count++;

#endif

        /* Check trace mode exceptions */

        if (unlikely(ctx.singlestep_enabled & CPU_SINGLE_STEP &&

                     (ctx.nip <= 0x100 || ctx.nip > 0xF00) &&

                     ctx.exception != POWERPC_SYSCALL &&

                     ctx.exception != POWERPC_EXCP_TRAP &&

                     ctx.exception != POWERPC_EXCP_BRANCH)) {

            gen_exception(ctxp, POWERPC_EXCP_TRACE);

        } else if (unlikely(((ctx.nip & (TARGET_PAGE_SIZE - 1)) == 0) ||

                            (cs->singlestep_enabled) ||

                            singlestep ||

                            num_insns >= max_insns)) {

            /* if we reach a page boundary or are single stepping, stop

             * generation

             */

            break;

        }

        if (tcg_check_temp_count()) {

            fprintf(stderr, "Opcode %02x %02x %02x (%08x) leaked temporaries\n",

                    opc1(ctx.opcode), opc2(ctx.opcode), opc3(ctx.opcode),

                    ctx.opcode);

            exit(1);

        }

    }

    if (tb->cflags & CF_LAST_IO)

        gen_io_end();

    if (ctx.exception == POWERPC_EXCP_NONE) {

        gen_goto_tb(&ctx, 0, ctx.nip);

    } else if (ctx.exception != POWERPC_EXCP_BRANCH) {

        if (unlikely(cs->singlestep_enabled)) {

            gen_debug_exception(ctxp);

        }

        /* Generate the return instruction */

        tcg_gen_exit_tb(0);

    }

    gen_tb_end(tb, num_insns);



    tb->size = ctx.nip - pc_start;

    tb->icount = num_insns;



#if defined(DEBUG_DISAS)

    if (qemu_loglevel_mask(CPU_LOG_TB_IN_ASM)

        && qemu_log_in_addr_range(pc_start)) {

        int flags;

        flags = env->bfd_mach;

        flags |= ctx.le_mode << 16;

        qemu_log("IN: %s\n", lookup_symbol(pc_start));

        log_target_disas(cs, pc_start, ctx.nip - pc_start, flags);

        qemu_log("\n");

    }

#endif

}
