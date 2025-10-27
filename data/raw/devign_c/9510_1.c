static int qemu_signalfd_compat(const sigset_t *mask)

{

    pthread_attr_t attr;

    pthread_t tid;

    struct sigfd_compat_info *info;

    int fds[2];



    info = malloc(sizeof(*info));

    if (info == NULL) {

        errno = ENOMEM;

        return -1;

    }



    if (pipe(fds) == -1) {

        free(info);

        return -1;

    }



    memcpy(&info->mask, mask, sizeof(*mask));

    info->fd = fds[1];



    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);



    pthread_create(&tid, &attr, sigwait_compat, info);



    pthread_attr_destroy(&attr);



    return fds[0];

}
