int cpu_x86_exec(CPUX86State *env1)

{

    int saved_T0, saved_T1, saved_A0;

    CPUX86State *saved_env;

#ifdef reg_EAX

    int saved_EAX;

#endif

#ifdef reg_ECX

    int saved_ECX;

#endif

#ifdef reg_EDX

    int saved_EDX;

#endif

#ifdef reg_EBX

    int saved_EBX;

#endif

#ifdef reg_ESP

    int saved_ESP;

#endif

#ifdef reg_EBP

    int saved_EBP;

#endif

#ifdef reg_ESI

    int saved_ESI;

#endif

#ifdef reg_EDI

    int saved_EDI;

#endif

#ifdef __sparc__

    int saved_i7, tmp_T0;

#endif

    int code_gen_size, ret;

    void (*gen_func)(void);

    TranslationBlock *tb, **ptb;

    uint8_t *tc_ptr, *cs_base, *pc;

    unsigned int flags;



    /* first we save global registers */

    saved_T0 = T0;

    saved_T1 = T1;

    saved_A0 = A0;

    saved_env = env;

    env = env1;

#ifdef reg_EAX

    saved_EAX = EAX;

    EAX = env->regs[R_EAX];

#endif

#ifdef reg_ECX

    saved_ECX = ECX;

    ECX = env->regs[R_ECX];

#endif

#ifdef reg_EDX

    saved_EDX = EDX;

    EDX = env->regs[R_EDX];

#endif

#ifdef reg_EBX

    saved_EBX = EBX;

    EBX = env->regs[R_EBX];

#endif

#ifdef reg_ESP

    saved_ESP = ESP;

    ESP = env->regs[R_ESP];

#endif

#ifdef reg_EBP

    saved_EBP = EBP;

    EBP = env->regs[R_EBP];

#endif

#ifdef reg_ESI

    saved_ESI = ESI;

    ESI = env->regs[R_ESI];

#endif

#ifdef reg_EDI

    saved_EDI = EDI;

    EDI = env->regs[R_EDI];

#endif

#ifdef __sparc__

    /* we also save i7 because longjmp may not restore it */

    asm volatile ("mov %%i7, %0" : "=r" (saved_i7));

#endif

    

    /* put eflags in CPU temporary format */

    CC_SRC = env->eflags & (CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

    DF = 1 - (2 * ((env->eflags >> 10) & 1));

    CC_OP = CC_OP_EFLAGS;

    env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

    env->interrupt_request = 0;



    /* prepare setjmp context for exception handling */

    if (setjmp(env->jmp_env) == 0) {

        T0 = 0; /* force lookup of first TB */

        for(;;) {

#ifdef __sparc__

	  /* g1 can be modified by some libc? functions */ 

	    tmp_T0 = T0;

#endif	    

            if (env->interrupt_request) {

                env->exception_index = EXCP_INTERRUPT;

                cpu_loop_exit();

            }

#ifdef DEBUG_EXEC

            if (loglevel) {

                /* XXX: save all volatile state in cpu state */

                /* restore flags in standard format */

                env->regs[R_EAX] = EAX;

                env->regs[R_EBX] = EBX;

                env->regs[R_ECX] = ECX;

                env->regs[R_EDX] = EDX;

                env->regs[R_ESI] = ESI;

                env->regs[R_EDI] = EDI;

                env->regs[R_EBP] = EBP;

                env->regs[R_ESP] = ESP;

                env->eflags = env->eflags | cc_table[CC_OP].compute_all() | (DF & DF_MASK);

                cpu_x86_dump_state(env, logfile, 0);

                env->eflags &= ~(DF_MASK | CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C);

            }

#endif

            /* we compute the CPU state. We assume it will not

               change during the whole generated block. */

            flags = env->seg_cache[R_CS].seg_32bit << GEN_FLAG_CODE32_SHIFT;

            flags |= env->seg_cache[R_SS].seg_32bit << GEN_FLAG_SS32_SHIFT;

            flags |= (((unsigned long)env->seg_cache[R_DS].base | 

                       (unsigned long)env->seg_cache[R_ES].base |

                       (unsigned long)env->seg_cache[R_SS].base) != 0) << 

                GEN_FLAG_ADDSEG_SHIFT;

            if (!(env->eflags & VM_MASK)) {

                flags |= (env->segs[R_CS] & 3) << GEN_FLAG_CPL_SHIFT;

            } else {

                /* NOTE: a dummy CPL is kept */

                flags |= (1 << GEN_FLAG_VM_SHIFT);

                flags |= (3 << GEN_FLAG_CPL_SHIFT);

            }

            flags |= (env->eflags & (IOPL_MASK | TF_MASK));

            cs_base = env->seg_cache[R_CS].base;

            pc = cs_base + env->eip;

            tb = tb_find(&ptb, (unsigned long)pc, (unsigned long)cs_base, 

                         flags);

            if (!tb) {

                spin_lock(&tb_lock);

                /* if no translated code available, then translate it now */

                tb = tb_alloc((unsigned long)pc);

                if (!tb) {

                    /* flush must be done */

                    tb_flush();

                    /* cannot fail at this point */

                    tb = tb_alloc((unsigned long)pc);

                    /* don't forget to invalidate previous TB info */

                    ptb = &tb_hash[tb_hash_func((unsigned long)pc)];

                    T0 = 0;

                }

                tc_ptr = code_gen_ptr;

                tb->tc_ptr = tc_ptr;

                tb->cs_base = (unsigned long)cs_base;

                tb->flags = flags;

                ret = cpu_x86_gen_code(tb, CODE_GEN_MAX_SIZE, &code_gen_size);

                /* if invalid instruction, signal it */

                if (ret != 0) {

                    /* NOTE: the tb is allocated but not linked, so we

                       can leave it */

                    spin_unlock(&tb_lock);

                    raise_exception(EXCP06_ILLOP);

                }

                *ptb = tb;

                tb->hash_next = NULL;

                tb_link(tb);

                code_gen_ptr = (void *)(((unsigned long)code_gen_ptr + code_gen_size + CODE_GEN_ALIGN - 1) & ~(CODE_GEN_ALIGN - 1));

                spin_unlock(&tb_lock);

            }

#ifdef DEBUG_EXEC

	    if (loglevel) {

		fprintf(logfile, "Trace 0x%08lx [0x%08lx] %s\n",

			(long)tb->tc_ptr, (long)tb->pc,

			lookup_symbol((void *)tb->pc));

	    }

#endif

#ifdef __sparc__

	    T0 = tmp_T0;

#endif	    

            /* see if we can patch the calling TB */

            if (T0 != 0 && !(env->eflags & TF_MASK)) {

                spin_lock(&tb_lock);

                tb_add_jump((TranslationBlock *)(T0 & ~3), T0 & 3, tb);

                spin_unlock(&tb_lock);

            }

            tc_ptr = tb->tc_ptr;



            /* execute the generated code */

            gen_func = (void *)tc_ptr;

#if defined(__sparc__)

	    __asm__ __volatile__("call	%0\n\t"

				 "mov	%%o7,%%i0"

				 : /* no outputs */

				 : "r" (gen_func) 

				 : "i0", "i1", "i2", "i3", "i4", "i5");

#elif defined(__arm__)

            asm volatile ("mov pc, %0\n\t"

                          ".global exec_loop\n\t"

                          "exec_loop:\n\t"

                          : /* no outputs */

                          : "r" (gen_func)

                          : "r1", "r2", "r3", "r8", "r9", "r10", "r12", "r14");

#else

            gen_func();

#endif

        }

    }

    ret = env->exception_index;



    /* restore flags in standard format */

    env->eflags = env->eflags | cc_table[CC_OP].compute_all() | (DF & DF_MASK);



    /* restore global registers */

#ifdef reg_EAX

    EAX = saved_EAX;

#endif

#ifdef reg_ECX

    ECX = saved_ECX;

#endif

#ifdef reg_EDX

    EDX = saved_EDX;

#endif

#ifdef reg_EBX

    EBX = saved_EBX;

#endif

#ifdef reg_ESP

    ESP = saved_ESP;

#endif

#ifdef reg_EBP

    EBP = saved_EBP;

#endif

#ifdef reg_ESI

    ESI = saved_ESI;

#endif

#ifdef reg_EDI

    EDI = saved_EDI;

#endif

#ifdef __sparc__

    asm volatile ("mov %0, %%i7" : : "r" (saved_i7));

#endif

    T0 = saved_T0;

    T1 = saved_T1;

    A0 = saved_A0;

    env = saved_env;

    return ret;

}
