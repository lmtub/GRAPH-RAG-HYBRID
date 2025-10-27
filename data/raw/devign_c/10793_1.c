static inline int handle_cpu_signal(uintptr_t pc, unsigned long address,

                                    int is_write, sigset_t *old_set)

{

    CPUState *cpu;

    CPUClass *cc;

    int ret;



#if defined(DEBUG_SIGNAL)

    printf("qemu: SIGSEGV pc=0x%08lx address=%08lx w=%d oldset=0x%08lx\n",

           pc, address, is_write, *(unsigned long *)old_set);

#endif

    /* XXX: locking issue */

    if (is_write && h2g_valid(address)) {

        switch (page_unprotect(h2g(address), pc)) {

        case 0:

            /* Fault not caused by a page marked unwritable to protect

             * cached translations, must be the guest binary's problem

             */

            break;

        case 1:

            /* Fault caused by protection of cached translation; TBs

             * invalidated, so resume execution

             */

            return 1;

        case 2:

            /* Fault caused by protection of cached translation, and the

             * currently executing TB was modified and must be exited

             * immediately.

             */

            cpu_exit_tb_from_sighandler(current_cpu, old_set);

            g_assert_not_reached();

        default:

            g_assert_not_reached();

        }

    }



    /* Convert forcefully to guest address space, invalid addresses

       are still valid segv ones */

    address = h2g_nocheck(address);



    cpu = current_cpu;

    cc = CPU_GET_CLASS(cpu);

    /* see if it is an MMU fault */

    g_assert(cc->handle_mmu_fault);

    ret = cc->handle_mmu_fault(cpu, address, is_write, MMU_USER_IDX);

    if (ret < 0) {

        return 0; /* not an MMU fault */

    }

    if (ret == 0) {

        return 1; /* the MMU fault was handled without causing real CPU fault */

    }



    /* Now we have a real cpu fault.  Since this is the exact location of

     * the exception, we must undo the adjustment done by cpu_restore_state

     * for handling call return addresses.  */

    cpu_restore_state(cpu, pc + GETPC_ADJ);



    sigprocmask(SIG_SETMASK, old_set, NULL);

    cpu_loop_exit(cpu);



    /* never comes here */

    return 1;

}
