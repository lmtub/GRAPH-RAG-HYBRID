static void host_signal_handler(int host_signum, siginfo_t *info,

                                void *puc)

{

    CPUArchState *env = thread_cpu->env_ptr;

    CPUState *cpu = ENV_GET_CPU(env);

    TaskState *ts = cpu->opaque;



    int sig;

    target_siginfo_t tinfo;

    ucontext_t *uc = puc;

    struct emulated_sigtable *k;



    /* the CPU emulator uses some host signals to detect exceptions,

       we forward to it some signals */

    if ((host_signum == SIGSEGV || host_signum == SIGBUS)

        && info->si_code > 0) {

        if (cpu_signal_handler(host_signum, info, puc))

            return;

    }



    /* get target signal number */

    sig = host_to_target_signal(host_signum);

    if (sig < 1 || sig > TARGET_NSIG)

        return;

    trace_user_host_signal(env, host_signum, sig);



    rewind_if_in_safe_syscall(puc);



    host_to_target_siginfo_noswap(&tinfo, info);

    k = &ts->sigtab[sig - 1];

    k->info = tinfo;

    k->pending = sig;

    ts->signal_pending = 1;



    /* Block host signals until target signal handler entered. We

     * can't block SIGSEGV or SIGBUS while we're executing guest

     * code in case the guest code provokes one in the window between

     * now and it getting out to the main loop. Signals will be

     * unblocked again in process_pending_signals().

     */

    sigfillset(&uc->uc_sigmask);

    sigdelset(&uc->uc_sigmask, SIGSEGV);

    sigdelset(&uc->uc_sigmask, SIGBUS);



    /* interrupt the virtual CPU as soon as possible */

    cpu_exit(thread_cpu);

}
