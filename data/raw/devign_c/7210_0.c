void os_mem_prealloc(int fd, char *area, size_t memory)

{

    int ret;

    struct sigaction act, oldact;

    sigset_t set, oldset;



    memset(&act, 0, sizeof(act));

    act.sa_handler = &sigbus_handler;

    act.sa_flags = 0;



    ret = sigaction(SIGBUS, &act, &oldact);

    if (ret) {

        perror("os_mem_prealloc: failed to install signal handler");

        exit(1);

    }



    /* unblock SIGBUS */

    sigemptyset(&set);

    sigaddset(&set, SIGBUS);

    pthread_sigmask(SIG_UNBLOCK, &set, &oldset);



    if (sigsetjmp(sigjump, 1)) {

        fprintf(stderr, "os_mem_prealloc: Insufficient free host memory "

                        "pages available to allocate guest RAM\n");

        exit(1);

    } else {

        int i;

        size_t hpagesize = fd_getpagesize(fd);

        size_t numpages = DIV_ROUND_UP(memory, hpagesize);



        /* MAP_POPULATE silently ignores failures */

        for (i = 0; i < numpages; i++) {

            memset(area + (hpagesize * i), 0, 1);

        }



        ret = sigaction(SIGBUS, &oldact, NULL);

        if (ret) {

            perror("os_mem_prealloc: failed to reinstall signal handler");

            exit(1);

        }



        pthread_sigmask(SIG_SETMASK, &oldset, NULL);

    }

}
