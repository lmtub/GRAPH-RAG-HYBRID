static void qemu_tcg_init_cpu_signals(void)

{

#ifdef CONFIG_IOTHREAD

    sigset_t set;

    struct sigaction sigact;



    memset(&sigact, 0, sizeof(sigact));

    sigact.sa_handler = cpu_signal;

    sigaction(SIG_IPI, &sigact, NULL);



    sigemptyset(&set);

    sigaddset(&set, SIG_IPI);

    pthread_sigmask(SIG_UNBLOCK, &set, NULL);

#endif

}
