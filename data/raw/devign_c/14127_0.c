static void QEMU_NORETURN force_sig(int sig)

{

    int host_sig;

    host_sig = target_to_host_signal(sig);

    fprintf(stderr, "qemu: uncaught target signal %d (%s) - exiting\n",

            sig, strsignal(host_sig));

#if 1

    gdb_signalled(thread_env, sig);

    _exit(-host_sig);

#else

    {

        struct sigaction act;

        sigemptyset(&act.sa_mask);

        act.sa_flags = SA_SIGINFO;

        act.sa_sigaction = SIG_DFL;

        sigaction(SIGABRT, &act, NULL);

        abort();

    }

#endif

}
