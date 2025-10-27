int cpu_exec(CPUState *cpu)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

#ifdef TARGET_I386

    X86CPU *x86_cpu = X86_CPU(cpu);

    CPUArchState *env = &x86_cpu->env;

#endif

    int ret, interrupt_request;

    TranslationBlock *tb;

    uint8_t *tc_ptr;

    uintptr_t next_tb;

    SyncClocks sc;



    if (cpu->halted) {

#if defined(TARGET_I386) && !defined(CONFIG_USER_ONLY)

        if (cpu->interrupt_request & CPU_INTERRUPT_POLL) {

            apic_poll_irq(x86_cpu->apic_state);

            cpu_reset_interrupt(cpu, CPU_INTERRUPT_POLL);

        }

#endif

        if (!cpu_has_work(cpu)) {

            return EXCP_HALTED;

        }



        cpu->halted = 0;

    }



    current_cpu = cpu;

    atomic_mb_set(&tcg_current_cpu, cpu);

    rcu_read_lock();



    if (unlikely(atomic_mb_read(&exit_request))) {

        cpu->exit_request = 1;

    }



    cc->cpu_exec_enter(cpu);



    /* Calculate difference between guest clock and host clock.

     * This delay includes the delay of the last cycle, so

     * what we have to do is sleep until it is 0. As for the

     * advance/delay we gain here, we try to fix it next time.

     */

    init_delay_params(&sc, cpu);



    /* prepare setjmp context for exception handling */

    for(;;) {

        if (sigsetjmp(cpu->jmp_env, 0) == 0) {

            /* if an exception is pending, we execute it here */

            if (cpu->exception_index >= 0) {

                if (cpu->exception_index >= EXCP_INTERRUPT) {

                    /* exit request from the cpu execution loop */

                    ret = cpu->exception_index;

                    if (ret == EXCP_DEBUG) {

                        cpu_handle_debug_exception(cpu);

                    }

                    cpu->exception_index = -1;

                    break;

                } else {

#if defined(CONFIG_USER_ONLY)

                    /* if user mode only, we simulate a fake exception

                       which will be handled outside the cpu execution

                       loop */

#if defined(TARGET_I386)

                    cc->do_interrupt(cpu);

#endif

                    ret = cpu->exception_index;

                    cpu->exception_index = -1;

                    break;

#else

                    cc->do_interrupt(cpu);

                    cpu->exception_index = -1;

#endif

                }

            }



            next_tb = 0; /* force lookup of first TB */

            for(;;) {

                interrupt_request = cpu->interrupt_request;

                if (unlikely(interrupt_request)) {

                    if (unlikely(cpu->singlestep_enabled & SSTEP_NOIRQ)) {

                        /* Mask out external interrupts for this step. */

                        interrupt_request &= ~CPU_INTERRUPT_SSTEP_MASK;

                    }

                    if (interrupt_request & CPU_INTERRUPT_DEBUG) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_DEBUG;

                        cpu->exception_index = EXCP_DEBUG;

                        cpu_loop_exit(cpu);

                    }

                    if (interrupt_request & CPU_INTERRUPT_HALT) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_HALT;

                        cpu->halted = 1;

                        cpu->exception_index = EXCP_HLT;

                        cpu_loop_exit(cpu);

                    }

#if defined(TARGET_I386)

                    if (interrupt_request & CPU_INTERRUPT_INIT) {

                        cpu_svm_check_intercept_param(env, SVM_EXIT_INIT, 0);

                        do_cpu_init(x86_cpu);

                        cpu->exception_index = EXCP_HALTED;

                        cpu_loop_exit(cpu);

                    }

#else

                    if (interrupt_request & CPU_INTERRUPT_RESET) {

                        cpu_reset(cpu);

                    }

#endif

                    /* The target hook has 3 exit conditions:

                       False when the interrupt isn't processed,

                       True when it is, and we should restart on a new TB,

                       and via longjmp via cpu_loop_exit.  */

                    if (cc->cpu_exec_interrupt(cpu, interrupt_request)) {

                        next_tb = 0;

                    }

                    /* Don't use the cached interrupt_request value,

                       do_interrupt may have updated the EXITTB flag. */

                    if (cpu->interrupt_request & CPU_INTERRUPT_EXITTB) {

                        cpu->interrupt_request &= ~CPU_INTERRUPT_EXITTB;

                        /* ensure that no TB jump will be modified as

                           the program flow was changed */

                        next_tb = 0;

                    }

                }

                if (unlikely(cpu->exit_request)) {

                    cpu->exit_request = 0;

                    cpu->exception_index = EXCP_INTERRUPT;

                    cpu_loop_exit(cpu);

                }

                tb_lock();

                tb = tb_find_fast(cpu);

                /* Note: we do it here to avoid a gcc bug on Mac OS X when

                   doing it in tb_find_slow */

                if (tcg_ctx.tb_ctx.tb_invalidated_flag) {

                    /* as some TB could have been invalidated because

                       of memory exceptions while generating the code, we

                       must recompute the hash index here */

                    next_tb = 0;

                    tcg_ctx.tb_ctx.tb_invalidated_flag = 0;

                }

                if (qemu_loglevel_mask(CPU_LOG_EXEC)) {

                    qemu_log("Trace %p [" TARGET_FMT_lx "] %s\n",

                             tb->tc_ptr, tb->pc, lookup_symbol(tb->pc));

                }

                /* see if we can patch the calling TB. When the TB

                   spans two pages, we cannot safely do a direct

                   jump. */

                if (next_tb != 0 && tb->page_addr[1] == -1

                    && !qemu_loglevel_mask(CPU_LOG_TB_NOCHAIN)) {

                    tb_add_jump((TranslationBlock *)(next_tb & ~TB_EXIT_MASK),

                                next_tb & TB_EXIT_MASK, tb);

                }

                tb_unlock();

                if (likely(!cpu->exit_request)) {

                    trace_exec_tb(tb, tb->pc);

                    tc_ptr = tb->tc_ptr;

                    /* execute the generated code */

                    cpu->current_tb = tb;

                    next_tb = cpu_tb_exec(cpu, tc_ptr);

                    cpu->current_tb = NULL;

                    switch (next_tb & TB_EXIT_MASK) {

                    case TB_EXIT_REQUESTED:

                        /* Something asked us to stop executing

                         * chained TBs; just continue round the main

                         * loop. Whatever requested the exit will also

                         * have set something else (eg exit_request or

                         * interrupt_request) which we will handle

                         * next time around the loop.  But we need to

                         * ensure the tcg_exit_req read in generated code

                         * comes before the next read of cpu->exit_request

                         * or cpu->interrupt_request.

                         */

                        smp_rmb();

                        next_tb = 0;

                        break;

                    case TB_EXIT_ICOUNT_EXPIRED:

                    {

                        /* Instruction counter expired.  */

                        int insns_left = cpu->icount_decr.u32;

                        if (cpu->icount_extra && insns_left >= 0) {

                            /* Refill decrementer and continue execution.  */

                            cpu->icount_extra += insns_left;

                            insns_left = MIN(0xffff, cpu->icount_extra);

                            cpu->icount_extra -= insns_left;

                            cpu->icount_decr.u16.low = insns_left;

                        } else {

                            if (insns_left > 0) {

                                /* Execute remaining instructions.  */

                                tb = (TranslationBlock *)(next_tb & ~TB_EXIT_MASK);

                                cpu_exec_nocache(cpu, insns_left, tb);

                                align_clocks(&sc, cpu);

                            }

                            cpu->exception_index = EXCP_INTERRUPT;

                            next_tb = 0;

                            cpu_loop_exit(cpu);

                        }

                        break;

                    }

                    default:

                        break;

                    }

                }

                /* Try to align the host and virtual clocks

                   if the guest is in advance */

                align_clocks(&sc, cpu);

                /* reset soft MMU for next block (it can currently

                   only be set by a memory fault) */

            } /* for(;;) */

        } else {

            /* Reload env after longjmp - the compiler may have smashed all

             * local variables as longjmp is marked 'noreturn'. */

            cpu = current_cpu;

            cc = CPU_GET_CLASS(cpu);

            cpu->can_do_io = 1;

#ifdef TARGET_I386

            x86_cpu = X86_CPU(cpu);

            env = &x86_cpu->env;

#endif

            tb_lock_reset();

        }

    } /* for(;;) */



    cc->cpu_exec_exit(cpu);

    rcu_read_unlock();



    /* fail safe : never use current_cpu outside cpu_exec() */

    current_cpu = NULL;



    /* Does not need atomic_mb_set because a spurious wakeup is okay.  */

    atomic_set(&tcg_current_cpu, NULL);

    return ret;

}
