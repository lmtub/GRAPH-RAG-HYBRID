static void sigfd_handler(void *opaque)

{

    int fd = (intptr_t)opaque;

    struct qemu_signalfd_siginfo info;

    struct sigaction action;

    ssize_t len;



    while (1) {

        do {

            len = read(fd, &info, sizeof(info));

        } while (len == -1 && errno == EINTR);



        if (len == -1 && errno == EAGAIN) {

            break;

        }



        if (len != sizeof(info)) {

            printf("read from sigfd returned %zd: %m\n", len);

            return;

        }



        sigaction(info.ssi_signo, NULL, &action);

        if ((action.sa_flags & SA_SIGINFO) && action.sa_sigaction) {

            action.sa_sigaction(info.ssi_signo,

                                (siginfo_t *)&info, NULL);

        } else if (action.sa_handler) {

            action.sa_handler(info.ssi_signo);

        }

    }

}
