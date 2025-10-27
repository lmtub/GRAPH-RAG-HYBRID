int cpu_exec(CPUState *env)

{

    int ret, interrupt_request;

    TranslationBlock *tb;

    uint8_t *tc_ptr;

    unsigned long next_tb;



    if (env->halted) {

        if (!cpu_has_work(env)) {

            return EXCP_HALTED;

        }



        env->halted = 0;

    }



    cpu_single_env = env;



    if (unlikely(exit_request)) {

        env->exit_request = 1;

    }



#if defined(TARGET_I386)

    /* put eflags in CPU temporary format */

    CC_SRC = env->eflags & (CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

    DF = 1 - (2 * ((env->eflags >> 10) & 1));

    CC_OP = CC_OP_EFLAGS;

    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

#elif defined(TARGET_SPARC)

#elif defined(TARGET_M68K)

    env->cc_op = CC_OP_FLAGS;

    env->cc_dest = env->sr & 0xf;

    env->cc_x = (env->sr >> 4) & 1;

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_ARM)

#elif defined(TARGET_UNICORE32)

#elif defined(TARGET_PPC)

#elif defined(TARGET_LM32)

#elif defined(TARGET_MICROBLAZE)

#elif defined(TARGET_MIPS)

#elif defined(TARGET_SH4)

#elif defined(TARGET_CRIS)

#elif defined(TARGET_S390X)

    /* XXXXX */

#else

#error unsupported target CPU

#endif

    env->exception_index = -1;



    /* prepare setjmp context for exception handling */

    for(;;) {

        if (setjmp(env->jmp_env) == 0) {

            /* if an exception is pending, we execute it here */

            if (env->exception_index >= 0) {

                if (env->exception_index >= EXCP_INTERRUPT) {

                    /* exit request from the cpu execution loop */

                    ret = env->exception_index;

                    if (ret == EXCP_DEBUG) {

                        cpu_handle_debug_exception(env);

                    }

                    break;


#if defined(CONFIG_USER_ONLY)

                    /* if user mode only, we simulate a fake exception

                       which will be handled outside the cpu execution

                       loop */

#if defined(TARGET_I386)

                    do_interrupt(env);

#endif

                    ret = env->exception_index;

                    break;

#else

                    do_interrupt(env);

                    env->exception_index = -1;

#endif

                }

            }



            next_tb = 0; /* force lookup of first TB */

            for(;;) {

                interrupt_request = env->interrupt_request;

                if (unlikely(interrupt_request)) {

                    if (unlikely(env->singlestep_enabled & SSTEP_NOIRQ)) {

                        /* Mask out external interrupts for this step. */

                        interrupt_request &= ~CPU_INTERRUPT_SSTEP_MASK;

                    }

                    if (interrupt_request & CPU_INTERRUPT_DEBUG) {

                        env->interrupt_request &= ~CPU_INTERRUPT_DEBUG;

                        env->exception_index = EXCP_DEBUG;

                        cpu_loop_exit(env);

                    }

#if defined(TARGET_ARM) || defined(TARGET_SPARC) || defined(TARGET_MIPS) || \

    defined(TARGET_PPC) || defined(TARGET_ALPHA) || defined(TARGET_CRIS) || \

    defined(TARGET_MICROBLAZE) || defined(TARGET_LM32) || defined(TARGET_UNICORE32)

                    if (interrupt_request & CPU_INTERRUPT_HALT) {

                        env->interrupt_request &= ~CPU_INTERRUPT_HALT;

                        env->halted = 1;

                        env->exception_index = EXCP_HLT;

                        cpu_loop_exit(env);

                    }

#endif

#if defined(TARGET_I386)

                    if (interrupt_request & CPU_INTERRUPT_INIT) {

                            svm_check_intercept(env, SVM_EXIT_INIT);

                            do_cpu_init(env);

                            env->exception_index = EXCP_HALTED;

                            cpu_loop_exit(env);

                    } else if (interrupt_request & CPU_INTERRUPT_SIPI) {

                            do_cpu_sipi(env);

                    } else if (env->hflags2 & HF2_GIF_MASK) {

                        if ((interrupt_request & CPU_INTERRUPT_SMI) &&

                            !(env->hflags & HF_SMM_MASK)) {

                            svm_check_intercept(env, SVM_EXIT_SMI);

                            env->interrupt_request &= ~CPU_INTERRUPT_SMI;

                            do_smm_enter(env);

                            next_tb = 0;

                        } else if ((interrupt_request & CPU_INTERRUPT_NMI) &&

                                   !(env->hflags2 & HF2_NMI_MASK)) {

                            env->interrupt_request &= ~CPU_INTERRUPT_NMI;

                            env->hflags2 |= HF2_NMI_MASK;

                            do_interrupt_x86_hardirq(env, EXCP02_NMI, 1);

                            next_tb = 0;

			} else if (interrupt_request & CPU_INTERRUPT_MCE) {

                            env->interrupt_request &= ~CPU_INTERRUPT_MCE;

                            do_interrupt_x86_hardirq(env, EXCP12_MCHK, 0);

                            next_tb = 0;

                        } else if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                                   (((env->hflags2 & HF2_VINTR_MASK) && 

                                     (env->hflags2 & HF2_HIF_MASK)) ||

                                    (!(env->hflags2 & HF2_VINTR_MASK) && 

                                     (env->eflags & IF_MASK && 

                                      !(env->hflags & HF_INHIBIT_IRQ_MASK))))) {

                            int intno;

                            svm_check_intercept(env, SVM_EXIT_INTR);

                            env->interrupt_request &= ~(CPU_INTERRUPT_HARD | CPU_INTERRUPT_VIRQ);

                            intno = cpu_get_pic_interrupt(env);

                            qemu_log_mask(CPU_LOG_TB_IN_ASM, "Servicing hardware INT=0x%02x\n", intno);

                            do_interrupt_x86_hardirq(env, intno, 1);

                            /* ensure that no TB jump will be modified as

                               the program flow was changed */

                            next_tb = 0;

#if !defined(CONFIG_USER_ONLY)

                        } else if ((interrupt_request & CPU_INTERRUPT_VIRQ) &&

                                   (env->eflags & IF_MASK) && 

                                   !(env->hflags & HF_INHIBIT_IRQ_MASK)) {

                            int intno;

                            /* FIXME: this should respect TPR */

                            svm_check_intercept(env, SVM_EXIT_VINTR);

                            intno = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_vector));

                            qemu_log_mask(CPU_LOG_TB_IN_ASM, "Servicing virtual hardware INT=0x%02x\n", intno);

                            do_interrupt_x86_hardirq(env, intno, 1);

                            env->interrupt_request &= ~CPU_INTERRUPT_VIRQ;

                            next_tb = 0;

#endif

                        }

                    }

#elif defined(TARGET_PPC)

#if 0

                    if ((interrupt_request & CPU_INTERRUPT_RESET)) {

                        cpu_reset(env);

                    }

#endif

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        ppc_hw_interrupt(env);

                        if (env->pending_interrupts == 0)

                            env->interrupt_request &= ~CPU_INTERRUPT_HARD;

                        next_tb = 0;

                    }

#elif defined(TARGET_LM32)

                    if ((interrupt_request & CPU_INTERRUPT_HARD)

                        && (env->ie & IE_IE)) {

                        env->exception_index = EXCP_IRQ;

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_MICROBLAZE)

                    if ((interrupt_request & CPU_INTERRUPT_HARD)

                        && (env->sregs[SR_MSR] & MSR_IE)

                        && !(env->sregs[SR_MSR] & (MSR_EIP | MSR_BIP))

                        && !(env->iflags & (D_FLAG | IMM_FLAG))) {

                        env->exception_index = EXCP_IRQ;

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_MIPS)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        cpu_mips_hw_interrupts_pending(env)) {

                        /* Raise it */

                        env->exception_index = EXCP_EXT_INTERRUPT;

                        env->error_code = 0;

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_SPARC)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        if (cpu_interrupts_enabled(env) &&

                            env->interrupt_index > 0) {

                            int pil = env->interrupt_index & 0xf;

                            int type = env->interrupt_index & 0xf0;



                            if (((type == TT_EXTINT) &&

                                  cpu_pil_allowed(env, pil)) ||

                                  type != TT_EXTINT) {

                                env->exception_index = env->interrupt_index;

                                do_interrupt(env);

                                next_tb = 0;

                            }

                        }

		    }

#elif defined(TARGET_ARM)

                    if (interrupt_request & CPU_INTERRUPT_FIQ

                        && !(env->uncached_cpsr & CPSR_F)) {

                        env->exception_index = EXCP_FIQ;

                        do_interrupt(env);

                        next_tb = 0;

                    }

                    /* ARMv7-M interrupt return works by loading a magic value

                       into the PC.  On real hardware the load causes the

                       return to occur.  The qemu implementation performs the

                       jump normally, then does the exception return when the

                       CPU tries to execute code at the magic address.

                       This will cause the magic PC value to be pushed to

                       the stack if an interrupt occurred at the wrong time.

                       We avoid this by disabling interrupts when

                       pc contains a magic address.  */

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && ((IS_M(env) && env->regs[15] < 0xfffffff0)

                            || !(env->uncached_cpsr & CPSR_I))) {

                        env->exception_index = EXCP_IRQ;

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_UNICORE32)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && !(env->uncached_asr & ASR_I)) {

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_SH4)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_ALPHA)

                    {

                        int idx = -1;

                        /* ??? This hard-codes the OSF/1 interrupt levels.  */

		        switch (env->pal_mode ? 7 : env->ps & PS_INT_MASK) {

                        case 0 ... 3:

                            if (interrupt_request & CPU_INTERRUPT_HARD) {

                                idx = EXCP_DEV_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 4:

                            if (interrupt_request & CPU_INTERRUPT_TIMER) {

                                idx = EXCP_CLK_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 5:

                            if (interrupt_request & CPU_INTERRUPT_SMP) {

                                idx = EXCP_SMP_INTERRUPT;

                            }

                            /* FALLTHRU */

                        case 6:

                            if (interrupt_request & CPU_INTERRUPT_MCHK) {

                                idx = EXCP_MCHK;

                            }

                        }

                        if (idx >= 0) {

                            env->exception_index = idx;

                            env->error_code = 0;

                            do_interrupt(env);

                            next_tb = 0;

                        }

                    }

#elif defined(TARGET_CRIS)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && (env->pregs[PR_CCS] & I_FLAG)

                        && !env->locked_irq) {

                        env->exception_index = EXCP_IRQ;

                        do_interrupt(env);

                        next_tb = 0;

                    }

                    if (interrupt_request & CPU_INTERRUPT_NMI

                        && (env->pregs[PR_CCS] & M_FLAG)) {

                        env->exception_index = EXCP_NMI;

                        do_interrupt(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_M68K)

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && ((env->sr & SR_I) >> SR_I_SHIFT)

                            < env->pending_level) {

                        /* Real hardware gets the interrupt vector via an

                           IACK cycle at this point.  Current emulated

                           hardware doesn't rely on this, so we

                           provide/save the vector when the interrupt is

                           first signalled.  */

                        env->exception_index = env->pending_vector;

                        do_interrupt_m68k_hardirq(env);

                        next_tb = 0;

                    }

#elif defined(TARGET_S390X) && !defined(CONFIG_USER_ONLY)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        (env->psw.mask & PSW_MASK_EXT)) {

                        do_interrupt(env);

                        next_tb = 0;

                    }

#endif

                   /* Don't use the cached interrupt_request value,

                      do_interrupt may have updated the EXITTB flag. */

                    if (env->interrupt_request & CPU_INTERRUPT_EXITTB) {

                        env->interrupt_request &= ~CPU_INTERRUPT_EXITTB;

                        /* ensure that no TB jump will be modified as

                           the program flow was changed */

                        next_tb = 0;

                    }

                }

                if (unlikely(env->exit_request)) {

                    env->exit_request = 0;

                    env->exception_index = EXCP_INTERRUPT;

                    cpu_loop_exit(env);

                }

#if defined(DEBUG_DISAS) || defined(CONFIG_DEBUG_EXEC)

                if (qemu_loglevel_mask(CPU_LOG_TB_CPU)) {

                    /* restore flags in standard format */

#if defined(TARGET_I386)

                    env->eflags = env->eflags | cpu_cc_compute_all(env, CC_OP)

                        | (DF & DF_MASK);

                    log_cpu_state(env, X86_DUMP_CCOP);

                    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

#elif defined(TARGET_M68K)

                    cpu_m68k_flush_flags(env, env->cc_op);

                    env->cc_op = CC_OP_FLAGS;

                    env->sr = (env->sr & 0xffe0)

                              | env->cc_dest | (env->cc_x << 4);

                    log_cpu_state(env, 0);

#else

                    log_cpu_state(env, 0);

#endif

                }

#endif /* DEBUG_DISAS || CONFIG_DEBUG_EXEC */

                spin_lock(&tb_lock);

                tb = tb_find_fast(env);

                /* Note: we do it here to avoid a gcc bug on Mac OS X when

                   doing it in tb_find_slow */

                if (tb_invalidated_flag) {

                    /* as some TB could have been invalidated because

                       of memory exceptions while generating the code, we

                       must recompute the hash index here */

                    next_tb = 0;

                    tb_invalidated_flag = 0;

                }

#ifdef CONFIG_DEBUG_EXEC

                qemu_log_mask(CPU_LOG_EXEC, "Trace 0x%08lx [" TARGET_FMT_lx "] %s\n",

                             (long)tb->tc_ptr, tb->pc,

                             lookup_symbol(tb->pc));

#endif

                /* see if we can patch the calling TB. When the TB

                   spans two pages, we cannot safely do a direct

                   jump. */

                if (next_tb != 0 && tb->page_addr[1] == -1) {

                    tb_add_jump((TranslationBlock *)(next_tb & ~3), next_tb & 3, tb);

                }

                spin_unlock(&tb_lock);



                /* cpu_interrupt might be called while translating the

                   TB, but before it is linked into a potentially

                   infinite loop and becomes env->current_tb. Avoid

                   starting execution if there is a pending interrupt. */

                env->current_tb = tb;

                barrier();

                if (likely(!env->exit_request)) {

                    tc_ptr = tb->tc_ptr;

                /* execute the generated code */

                    next_tb = tcg_qemu_tb_exec(env, tc_ptr);

                    if ((next_tb & 3) == 2) {

                        /* Instruction counter expired.  */

                        int insns_left;

                        tb = (TranslationBlock *)(long)(next_tb & ~3);

                        /* Restore PC.  */

                        cpu_pc_from_tb(env, tb);

                        insns_left = env->icount_decr.u32;

                        if (env->icount_extra && insns_left >= 0) {

                            /* Refill decrementer and continue execution.  */

                            env->icount_extra += insns_left;

                            if (env->icount_extra > 0xffff) {

                                insns_left = 0xffff;


                                insns_left = env->icount_extra;

                            }

                            env->icount_extra -= insns_left;

                            env->icount_decr.u16.low = insns_left;


                            if (insns_left > 0) {

                                /* Execute remaining instructions.  */

                                cpu_exec_nocache(env, insns_left, tb);

                            }

                            env->exception_index = EXCP_INTERRUPT;

                            next_tb = 0;

                            cpu_loop_exit(env);

                        }

                    }

                }

                env->current_tb = NULL;

                /* reset soft MMU for next block (it can currently

                   only be set by a memory fault) */

            } /* for(;;) */





        }

    } /* for(;;) */





#if defined(TARGET_I386)

    /* restore flags in standard format */

    env->eflags = env->eflags | cpu_cc_compute_all(env, CC_OP)

        | (DF & DF_MASK);

#elif defined(TARGET_ARM)

    /* XXX: Save/restore host fpu exception state?.  */

#elif defined(TARGET_UNICORE32)

#elif defined(TARGET_SPARC)

#elif defined(TARGET_PPC)

#elif defined(TARGET_LM32)

#elif defined(TARGET_M68K)

    cpu_m68k_flush_flags(env, env->cc_op);

    env->cc_op = CC_OP_FLAGS;

    env->sr = (env->sr & 0xffe0)

              | env->cc_dest | (env->cc_x << 4);

#elif defined(TARGET_MICROBLAZE)

#elif defined(TARGET_MIPS)

#elif defined(TARGET_SH4)

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_CRIS)

#elif defined(TARGET_S390X)

    /* XXXXX */

#else

#error unsupported target CPU

#endif



    /* fail safe : never use cpu_single_env outside cpu_exec() */

    cpu_single_env = NULL;

    return ret;

}