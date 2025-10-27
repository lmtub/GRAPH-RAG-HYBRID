static inline int handle_cpu_signal(uintptr_t pc, siginfo_t *info,

                                    int is_write, sigset_t *old_set)

{

    CPUState *cpu = current_cpu;

    CPUClass *cc;

    int ret;

    unsigned long address = (unsigned long)info->si_addr;



    /* We must handle PC addresses from two different sources:

     * a call return address and a signal frame address.

     *

     * Within cpu_restore_state_from_tb we assume the former and adjust

     * the address by -GETPC_ADJ so that the address is within the call

     * insn so that addr does not accidentally match the beginning of the

     * next guest insn.

     *

     * However, when the PC comes from the signal frame, it points to

     * the actual faulting host insn and not a call insn.  Subtracting

     * GETPC_ADJ in that case may accidentally match the previous guest insn.

     *

     * So for the later case, adjust forward to compensate for what

     * will be done later by cpu_restore_state_from_tb.

     */

    if (helper_retaddr) {

        pc = helper_retaddr;

    } else {

        pc += GETPC_ADJ;

    }



    /* For synchronous signals we expect to be coming from the vCPU

     * thread (so current_cpu should be valid) and either from running

     * code or during translation which can fault as we cross pages.

     *

     * If neither is true then something has gone wrong and we should

     * abort rather than try and restart the vCPU execution.

     */

    if (!cpu || !cpu->running) {

        printf("qemu:%s received signal outside vCPU context @ pc=0x%"

               PRIxPTR "\n",  __func__, pc);

        abort();

    }



#if defined(DEBUG_SIGNAL)

    printf("qemu: SIGSEGV pc=0x%08lx address=%08lx w=%d oldset=0x%08lx\n",

           pc, address, is_write, *(unsigned long *)old_set);

#endif

    /* XXX: locking issue */

    if (is_write && h2g_valid(address)) {

        switch (page_unprotect(h2g(address), pc)) {

        case 0:

            /* Fault not caused by a page marked unwritable to protect

             * cached translations, must be the guest binary's problem.

             */

            break;

        case 1:

            /* Fault caused by protection of cached translation; TBs

             * invalidated, so resume execution.  Retain helper_retaddr

             * for a possible second fault.

             */

            return 1;

        case 2:

            /* Fault caused by protection of cached translation, and the

             * currently executing TB was modified and must be exited

             * immediately.  Clear helper_retaddr for next execution.

             */

            helper_retaddr = 0;

            cpu_exit_tb_from_sighandler(cpu, old_set);

            /* NORETURN */



        default:

            g_assert_not_reached();

        }

    }



    /* Convert forcefully to guest address space, invalid addresses

       are still valid segv ones */

    address = h2g_nocheck(address);



    cc = CPU_GET_CLASS(cpu);

    /* see if it is an MMU fault */

    g_assert(cc->handle_mmu_fault);

    ret = cc->handle_mmu_fault(cpu, address, is_write, MMU_USER_IDX);



    if (ret == 0) {

        /* The MMU fault was handled without causing real CPU fault.

         *  Retain helper_retaddr for a possible second fault.

         */

        return 1;

    }



    /* All other paths lead to cpu_exit; clear helper_retaddr

     * for next execution.

     */

    helper_retaddr = 0;



    if (ret < 0) {

        return 0; /* not an MMU fault */

    }



    /* Now we have a real cpu fault.  */

    cpu_restore_state(cpu, pc);



    sigprocmask(SIG_SETMASK, old_set, NULL);

    cpu_loop_exit(cpu);



    /* never comes here */

    return 1;

}
