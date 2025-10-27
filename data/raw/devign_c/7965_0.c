static inline int handle_cpu_signal(uintptr_t pc, unsigned long address,

                                    int is_write, sigset_t *old_set,

                                    void *puc)

{

    TranslationBlock *tb;

    int ret;



    if (cpu_single_env) {

        env = cpu_single_env; /* XXX: find a correct solution for multithread */

    }

#if defined(DEBUG_SIGNAL)

    qemu_printf("qemu: SIGSEGV pc=0x%08lx address=%08lx w=%d oldset=0x%08lx\n",

                pc, address, is_write, *(unsigned long *)old_set);

#endif

    /* XXX: locking issue */

    if (is_write && page_unprotect(h2g(address), pc, puc)) {

        return 1;

    }



    /* see if it is an MMU fault */

    ret = cpu_handle_mmu_fault(env, address, is_write, MMU_USER_IDX);

    if (ret < 0) {

        return 0; /* not an MMU fault */

    }

    if (ret == 0) {

        return 1; /* the MMU fault was handled without causing real CPU fault */

    }

    /* now we have a real cpu fault */

    tb = tb_find_pc(pc);

    if (tb) {

        /* the PC is inside the translated code. It means that we have

           a virtual CPU fault */

        cpu_restore_state(tb, env, pc);

    }



    /* we restore the process signal mask as the sigreturn should

       do it (XXX: use sigsetjmp) */

    sigprocmask(SIG_SETMASK, old_set, NULL);

    exception_action(env);



    /* never comes here */

    return 1;

}
