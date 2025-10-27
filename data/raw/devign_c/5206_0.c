static void block_io_signals(void)

{

    sigset_t set;

    struct sigaction sigact;



    sigemptyset(&set);

    sigaddset(&set, SIGUSR2);

    sigaddset(&set, SIGIO);

    sigaddset(&set, SIGALRM);

    sigaddset(&set, SIGCHLD);

    pthread_sigmask(SIG_BLOCK, &set, NULL);



    sigemptyset(&set);

    sigaddset(&set, SIGUSR1);

    pthread_sigmask(SIG_UNBLOCK, &set, NULL);



    memset(&sigact, 0, sizeof(sigact));

    sigact.sa_handler = cpu_signal;

    sigaction(SIGUSR1, &sigact, NULL);

}
