int paio_init(void)

{

    struct sigaction act;

    PosixAioState *s;

    int fds[2];

    int ret;



    if (posix_aio_state)

        return 0;



    s = qemu_malloc(sizeof(PosixAioState));



    sigfillset(&act.sa_mask);

    act.sa_flags = 0; /* do not restart syscalls to interrupt select() */

    act.sa_handler = aio_signal_handler;

    sigaction(SIGUSR2, &act, NULL);



    s->first_aio = NULL;

    if (pipe(fds) == -1) {

        fprintf(stderr, "failed to create pipe\n");

        return -1;

    }



    s->rfd = fds[0];

    s->wfd = fds[1];



    fcntl(s->rfd, F_SETFL, O_NONBLOCK);

    fcntl(s->wfd, F_SETFL, O_NONBLOCK);



    qemu_aio_set_fd_handler(s->rfd, posix_aio_read, NULL, posix_aio_flush,

        posix_aio_process_queue, s);



    ret = pthread_attr_init(&attr);

    if (ret)

        die2(ret, "pthread_attr_init");



    ret = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    if (ret)

        die2(ret, "pthread_attr_setdetachstate");



    QTAILQ_INIT(&request_list);



    posix_aio_state = s;

    return 0;

}
