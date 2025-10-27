static inline int handle_cpu_signal(unsigned long pc, unsigned long address,

                                    int is_write, sigset_t *old_set,

                                    void *puc)

{

    TranslationBlock *tb;

    int ret;



    if (cpu_single_env)

        env = cpu_single_env; /* XXX: find a correct solution for multithread */

#if defined(DEBUG_SIGNAL)

    printf("qemu: SIGSEGV pc=0x%08lx address=%08lx w=%d oldset=0x%08lx\n",

           pc, address, is_write, *(unsigned long *)old_set);

#endif

    /* XXX: locking issue */

    if (is_write && page_unprotect(h2g(address), pc, puc)) {

        return 1;

    }



    /* see if it is an MMU fault */

    ret = cpu_mb_handle_mmu_fault(env, address, is_write, MMU_USER_IDX, 0);

    if (ret < 0)

        return 0; /* not an MMU fault */

    if (ret == 0)

        return 1; /* the MMU fault was handled without causing real CPU fault */



    /* now we have a real cpu fault */

    tb = tb_find_pc(pc);

    if (tb) {

        /* the PC is inside the translated code. It means that we have

           a virtual CPU fault */

        cpu_restore_state(tb, env, pc, puc);

    }

    if (ret == 1) {

#if 0

        printf("PF exception: PC=0x" TARGET_FMT_lx " error=0x%x %p\n",

               env->PC, env->error_code, tb);

#endif

    /* we restore the process signal mask as the sigreturn should

       do it (XXX: use sigsetjmp) */

        sigprocmask(SIG_SETMASK, old_set, NULL);

        cpu_loop_exit();

    } else {

        /* activate soft MMU for this block */

        cpu_resume_from_signal(env, puc);

    }

    /* never comes here */

    return 1;

}
