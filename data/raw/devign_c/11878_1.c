int cpu_exec(CPUState *env1)

{

#define DECLARE_HOST_REGS 1

#include "hostregs_helper.h"

#if defined(TARGET_SPARC)

#if defined(reg_REGWPTR)

    uint32_t *saved_regwptr;

#endif

#endif

    int ret, interrupt_request;

    long (*gen_func)(void);

    TranslationBlock *tb;

    uint8_t *tc_ptr;



    if (cpu_halted(env1) == EXCP_HALTED)

        return EXCP_HALTED;



    cpu_single_env = env1;



    /* first we save global registers */

#define SAVE_HOST_REGS 1

#include "hostregs_helper.h"

    env = env1;

    SAVE_GLOBALS();



    env_to_regs();

#if defined(TARGET_I386)

    /* put eflags in CPU temporary format */

    CC_SRC = env->eflags & (CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

    DF = 1 - (2 * ((env->eflags >> 10) & 1));

    CC_OP = CC_OP_EFLAGS;

    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

#elif defined(TARGET_SPARC)

#if defined(reg_REGWPTR)

    saved_regwptr = REGWPTR;

#endif

#elif defined(TARGET_M68K)

    env->cc_op = CC_OP_FLAGS;

    env->cc_dest = env->sr & 0xf;

    env->cc_x = (env->sr >> 4) & 1;

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_ARM)

#elif defined(TARGET_PPC)

#elif defined(TARGET_MIPS)

#elif defined(TARGET_SH4)

#elif defined(TARGET_CRIS)

    /* XXXXX */

#else

#error unsupported target CPU

#endif

    env->exception_index = -1;



    /* prepare setjmp context for exception handling */

    for(;;) {

        if (setjmp(env->jmp_env) == 0) {

            env->current_tb = NULL;

            /* if an exception is pending, we execute it here */

            if (env->exception_index >= 0) {

                if (env->exception_index >= EXCP_INTERRUPT) {

                    /* exit request from the cpu execution loop */

                    ret = env->exception_index;

                    break;

                } else if (env->user_mode_only) {

                    /* if user mode only, we simulate a fake exception

                       which will be handled outside the cpu execution

                       loop */

#if defined(TARGET_I386)

                    do_interrupt_user(env->exception_index,

                                      env->exception_is_int,

                                      env->error_code,

                                      env->exception_next_eip);

#endif

                    ret = env->exception_index;

                    break;

                } else {

#if defined(TARGET_I386)

                    /* simulate a real cpu exception. On i386, it can

                       trigger new exceptions, but we do not handle

                       double or triple faults yet. */

                    do_interrupt(env->exception_index,

                                 env->exception_is_int,

                                 env->error_code,

                                 env->exception_next_eip, 0);

                    /* successfully delivered */

                    env->old_exception = -1;

#elif defined(TARGET_PPC)

                    do_interrupt(env);

#elif defined(TARGET_MIPS)

                    do_interrupt(env);

#elif defined(TARGET_SPARC)

                    do_interrupt(env->exception_index);

#elif defined(TARGET_ARM)

                    do_interrupt(env);

#elif defined(TARGET_SH4)

		    do_interrupt(env);

#elif defined(TARGET_ALPHA)

                    do_interrupt(env);

#elif defined(TARGET_CRIS)

                    do_interrupt(env);

#elif defined(TARGET_M68K)

                    do_interrupt(0);

#endif

                }

                env->exception_index = -1;

            }

#ifdef USE_KQEMU

            if (kqemu_is_ok(env) && env->interrupt_request == 0) {

                int ret;

                env->eflags = env->eflags | cc_table[CC_OP].compute_all() | (DF & DF_MASK);

                ret = kqemu_cpu_exec(env);

                /* put eflags in CPU temporary format */

                CC_SRC = env->eflags & (CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

                DF = 1 - (2 * ((env->eflags >> 10) & 1));

                CC_OP = CC_OP_EFLAGS;

                env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

                if (ret == 1) {

                    /* exception */

                    longjmp(env->jmp_env, 1);

                } else if (ret == 2) {

                    /* softmmu execution needed */

                } else {

                    if (env->interrupt_request != 0) {

                        /* hardware interrupt will be executed just after */

                    } else {

                        /* otherwise, we restart */

                        longjmp(env->jmp_env, 1);

                    }

                }

            }

#endif



            T0 = 0; /* force lookup of first TB */

            for(;;) {

                SAVE_GLOBALS();

                interrupt_request = env->interrupt_request;

                if (__builtin_expect(interrupt_request, 0)

#if defined(TARGET_I386)

			&& env->hflags & HF_GIF_MASK

#endif

				) {

                    if (interrupt_request & CPU_INTERRUPT_DEBUG) {

                        env->interrupt_request &= ~CPU_INTERRUPT_DEBUG;

                        env->exception_index = EXCP_DEBUG;

                        cpu_loop_exit();

                    }

#if defined(TARGET_ARM) || defined(TARGET_SPARC) || defined(TARGET_MIPS) || \

    defined(TARGET_PPC) || defined(TARGET_ALPHA) || defined(TARGET_CRIS)

                    if (interrupt_request & CPU_INTERRUPT_HALT) {

                        env->interrupt_request &= ~CPU_INTERRUPT_HALT;

                        env->halted = 1;

                        env->exception_index = EXCP_HLT;

                        cpu_loop_exit();

                    }

#endif

#if defined(TARGET_I386)

                    if ((interrupt_request & CPU_INTERRUPT_SMI) &&

                        !(env->hflags & HF_SMM_MASK)) {

                        svm_check_intercept(SVM_EXIT_SMI);

                        env->interrupt_request &= ~CPU_INTERRUPT_SMI;

                        do_smm_enter();

                        BREAK_CHAIN;

                    } else if ((interrupt_request & CPU_INTERRUPT_NMI) &&

                        !(env->hflags & HF_NMI_MASK)) {

                        env->interrupt_request &= ~CPU_INTERRUPT_NMI;

                        env->hflags |= HF_NMI_MASK;

                        do_interrupt(EXCP02_NMI, 0, 0, 0, 1);

                        BREAK_CHAIN;

                    } else if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        (env->eflags & IF_MASK || env->hflags & HF_HIF_MASK) &&

                        !(env->hflags & HF_INHIBIT_IRQ_MASK)) {

                        int intno;

                        svm_check_intercept(SVM_EXIT_INTR);

                        env->interrupt_request &= ~(CPU_INTERRUPT_HARD | CPU_INTERRUPT_VIRQ);

                        intno = cpu_get_pic_interrupt(env);

                        if (loglevel & CPU_LOG_TB_IN_ASM) {

                            fprintf(logfile, "Servicing hardware INT=0x%02x\n", intno);

                        }

                        do_interrupt(intno, 0, 0, 0, 1);

                        /* ensure that no TB jump will be modified as

                           the program flow was changed */

                        BREAK_CHAIN;

#if !defined(CONFIG_USER_ONLY)

                    } else if ((interrupt_request & CPU_INTERRUPT_VIRQ) &&

                        (env->eflags & IF_MASK) && !(env->hflags & HF_INHIBIT_IRQ_MASK)) {

                         int intno;

                         /* FIXME: this should respect TPR */

                         env->interrupt_request &= ~CPU_INTERRUPT_VIRQ;

                         svm_check_intercept(SVM_EXIT_VINTR);

                         intno = ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_vector));

                         if (loglevel & CPU_LOG_TB_IN_ASM)

                             fprintf(logfile, "Servicing virtual hardware INT=0x%02x\n", intno);

	                 do_interrupt(intno, 0, 0, -1, 1);

                         stl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_ctl),

                                  ldl_phys(env->vm_vmcb + offsetof(struct vmcb, control.int_ctl)) & ~V_IRQ_MASK);

                        BREAK_CHAIN;

#endif

                    }

#elif defined(TARGET_PPC)

#if 0

                    if ((interrupt_request & CPU_INTERRUPT_RESET)) {

                        cpu_ppc_reset(env);

                    }

#endif

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        ppc_hw_interrupt(env);

                        if (env->pending_interrupts == 0)

                            env->interrupt_request &= ~CPU_INTERRUPT_HARD;

                        BREAK_CHAIN;

                    }

#elif defined(TARGET_MIPS)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

                        (env->CP0_Status & env->CP0_Cause & CP0Ca_IP_mask) &&

                        (env->CP0_Status & (1 << CP0St_IE)) &&

                        !(env->CP0_Status & (1 << CP0St_EXL)) &&

                        !(env->CP0_Status & (1 << CP0St_ERL)) &&

                        !(env->hflags & MIPS_HFLAG_DM)) {

                        /* Raise it */

                        env->exception_index = EXCP_EXT_INTERRUPT;

                        env->error_code = 0;

                        do_interrupt(env);

                        BREAK_CHAIN;

                    }

#elif defined(TARGET_SPARC)

                    if ((interrupt_request & CPU_INTERRUPT_HARD) &&

			(env->psret != 0)) {

			int pil = env->interrupt_index & 15;

			int type = env->interrupt_index & 0xf0;



			if (((type == TT_EXTINT) &&

			     (pil == 15 || pil > env->psrpil)) ||

			    type != TT_EXTINT) {

			    env->interrupt_request &= ~CPU_INTERRUPT_HARD;

			    do_interrupt(env->interrupt_index);

			    env->interrupt_index = 0;

#if !defined(TARGET_SPARC64) && !defined(CONFIG_USER_ONLY)

                            cpu_check_irqs(env);

#endif

                        BREAK_CHAIN;

			}

		    } else if (interrupt_request & CPU_INTERRUPT_TIMER) {

			//do_interrupt(0, 0, 0, 0, 0);

			env->interrupt_request &= ~CPU_INTERRUPT_TIMER;

		    }

#elif defined(TARGET_ARM)

                    if (interrupt_request & CPU_INTERRUPT_FIQ

                        && !(env->uncached_cpsr & CPSR_F)) {

                        env->exception_index = EXCP_FIQ;

                        do_interrupt(env);

                        BREAK_CHAIN;

                    }

                    /* ARMv7-M interrupt return works by loading a magic value

                       into the PC.  On real hardware the load causes the

                       return to occur.  The qemu implementation performs the

                       jump normally, then does the exception return when the

                       CPU tries to execute code at the magic address.

                       This will cause the magic PC value to be pushed to

                       the stack if an interrupt occured at the wrong time.

                       We avoid this by disabling interrupts when

                       pc contains a magic address.  */

                    if (interrupt_request & CPU_INTERRUPT_HARD

                        && ((IS_M(env) && env->regs[15] < 0xfffffff0)

                            || !(env->uncached_cpsr & CPSR_I))) {

                        env->exception_index = EXCP_IRQ;

                        do_interrupt(env);

                        BREAK_CHAIN;

                    }

#elif defined(TARGET_SH4)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        do_interrupt(env);

                        BREAK_CHAIN;

                    }

#elif defined(TARGET_ALPHA)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        do_interrupt(env);

                        BREAK_CHAIN;

                    }

#elif defined(TARGET_CRIS)

                    if (interrupt_request & CPU_INTERRUPT_HARD) {

                        do_interrupt(env);

                        BREAK_CHAIN;

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

                        do_interrupt(1);

                        BREAK_CHAIN;

                    }

#endif

                   /* Don't use the cached interupt_request value,

                      do_interrupt may have updated the EXITTB flag. */

                    if (env->interrupt_request & CPU_INTERRUPT_EXITTB) {

                        env->interrupt_request &= ~CPU_INTERRUPT_EXITTB;

                        /* ensure that no TB jump will be modified as

                           the program flow was changed */

                        BREAK_CHAIN;

                    }

                    if (interrupt_request & CPU_INTERRUPT_EXIT) {

                        env->interrupt_request &= ~CPU_INTERRUPT_EXIT;

                        env->exception_index = EXCP_INTERRUPT;

                        cpu_loop_exit();

                    }

                }

#ifdef DEBUG_EXEC

                if ((loglevel & CPU_LOG_TB_CPU)) {

                    /* restore flags in standard format */

                    regs_to_env();

#if defined(TARGET_I386)

                    env->eflags = env->eflags | cc_table[CC_OP].compute_all() | (DF & DF_MASK);

                    cpu_dump_state(env, logfile, fprintf, X86_DUMP_CCOP);

                    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

#elif defined(TARGET_ARM)

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_SPARC)

		    REGWPTR = env->regbase + (env->cwp * 16);

		    env->regwptr = REGWPTR;

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_PPC)

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_M68K)

                    cpu_m68k_flush_flags(env, env->cc_op);

                    env->cc_op = CC_OP_FLAGS;

                    env->sr = (env->sr & 0xffe0)

                              | env->cc_dest | (env->cc_x << 4);

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_MIPS)

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_SH4)

		    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_ALPHA)

                    cpu_dump_state(env, logfile, fprintf, 0);

#elif defined(TARGET_CRIS)

                    cpu_dump_state(env, logfile, fprintf, 0);

#else

#error unsupported target CPU

#endif

                }

#endif

                tb = tb_find_fast();

#ifdef DEBUG_EXEC

                if ((loglevel & CPU_LOG_EXEC)) {

                    fprintf(logfile, "Trace 0x%08lx [" TARGET_FMT_lx "] %s\n",

                            (long)tb->tc_ptr, tb->pc,

                            lookup_symbol(tb->pc));

                }

#endif

                RESTORE_GLOBALS();

                /* see if we can patch the calling TB. When the TB

                   spans two pages, we cannot safely do a direct

                   jump. */

                {

                    if (T0 != 0 &&

#if USE_KQEMU

                        (env->kqemu_enabled != 2) &&

#endif

                        tb->page_addr[1] == -1) {

                    spin_lock(&tb_lock);

                    tb_add_jump((TranslationBlock *)(long)(T0 & ~3), T0 & 3, tb);

                    spin_unlock(&tb_lock);

                }

                }

                tc_ptr = tb->tc_ptr;

                env->current_tb = tb;

                /* execute the generated code */

                gen_func = (void *)tc_ptr;

#if defined(__sparc__)

                __asm__ __volatile__("call	%0\n\t"

                                     "mov	%%o7,%%i0"

                                     : /* no outputs */

                                     : "r" (gen_func)

                                     : "i0", "i1", "i2", "i3", "i4", "i5",

                                       "o0", "o1", "o2", "o3", "o4", "o5",

                                       "l0", "l1", "l2", "l3", "l4", "l5",

                                       "l6", "l7");

#elif defined(__hppa__)

                asm volatile ("ble  0(%%sr4,%1)\n"

                              "copy %%r31,%%r18\n"

                              "copy %%r28,%0\n"

                              : "=r" (T0)

                              : "r" (gen_func)

                              : "r1", "r2", "r3", "r4", "r5", "r6", "r7",

                                "r8", "r9", "r10", "r11", "r12", "r13",

                                "r18", "r19", "r20", "r21", "r22", "r23",

                                "r24", "r25", "r26", "r27", "r28", "r29",

                                "r30", "r31");

#elif defined(__arm__)

                asm volatile ("mov pc, %0\n\t"

                              ".global exec_loop\n\t"

                              "exec_loop:\n\t"

                              : /* no outputs */

                              : "r" (gen_func)

                              : "r1", "r2", "r3", "r8", "r9", "r10", "r12", "r14");

#elif defined(__ia64)

		struct fptr {

			void *ip;

			void *gp;

		} fp;



		fp.ip = tc_ptr;

		fp.gp = code_gen_buffer + 2 * (1 << 20);

		(*(void (*)(void)) &fp)();

#else

                T0 = gen_func();

#endif

                env->current_tb = NULL;

                /* reset soft MMU for next block (it can currently

                   only be set by a memory fault) */

#if defined(TARGET_I386) && !defined(CONFIG_SOFTMMU)

                if (env->hflags & HF_SOFTMMU_MASK) {

                    env->hflags &= ~HF_SOFTMMU_MASK;

                    /* do not allow linking to another block */

                    T0 = 0;

                }

#endif

#if defined(USE_KQEMU)

#define MIN_CYCLE_BEFORE_SWITCH (100 * 1000)

                if (kqemu_is_ok(env) &&

                    (cpu_get_time_fast() - env->last_io_time) >= MIN_CYCLE_BEFORE_SWITCH) {

                    cpu_loop_exit();

                }

#endif

            } /* for(;;) */

        } else {

            env_to_regs();

        }

    } /* for(;;) */





#if defined(TARGET_I386)

    /* restore flags in standard format */

    env->eflags = env->eflags | cc_table[CC_OP].compute_all() | (DF & DF_MASK);

#elif defined(TARGET_ARM)

    /* XXX: Save/restore host fpu exception state?.  */

#elif defined(TARGET_SPARC)

#if defined(reg_REGWPTR)

    REGWPTR = saved_regwptr;

#endif

#elif defined(TARGET_PPC)

#elif defined(TARGET_M68K)

    cpu_m68k_flush_flags(env, env->cc_op);

    env->cc_op = CC_OP_FLAGS;

    env->sr = (env->sr & 0xffe0)

              | env->cc_dest | (env->cc_x << 4);

#elif defined(TARGET_MIPS)

#elif defined(TARGET_SH4)

#elif defined(TARGET_ALPHA)

#elif defined(TARGET_CRIS)

    /* XXXXX */

#else

#error unsupported target CPU

#endif



    /* restore global registers */

    RESTORE_GLOBALS();

#include "hostregs_helper.h"



    /* fail safe : never use cpu_single_env outside cpu_exec() */

    cpu_single_env = NULL;

    return ret;

}
