static void host_signal_handler(int host_signum, siginfo_t *info, 

                                void *puc)

{

    int sig;

    target_siginfo_t tinfo;



    /* the CPU emulator uses some host signals to detect exceptions,

       we we forward to it some signals */

    if (host_signum == SIGSEGV || host_signum == SIGBUS) {

        if (cpu_signal_handler(host_signum, info, puc))

            return;

    }



    /* get target signal number */

    sig = host_to_target_signal(host_signum);

    if (sig < 1 || sig > TARGET_NSIG)

        return;

#if defined(DEBUG_SIGNAL)

    fprintf(stderr, "qemu: got signal %d\n", sig);

    dump_regs(puc);

#endif

    host_to_target_siginfo_noswap(&tinfo, info);

    if (queue_signal(sig, &tinfo) == 1) {

        /* interrupt the virtual CPU as soon as possible */

        cpu_interrupt(global_env, CPU_INTERRUPT_EXIT);

    }

}
