int qemu_eventfd(int *fds)

{

#if defined(SYS_eventfd)

    int ret;



    ret = syscall(SYS_eventfd, 0);

    if (ret >= 0) {

        fds[0] = fds[1] = ret;

        return 0;

    } else if (!(ret == -1 && errno == ENOSYS))

        return ret;

#endif



    return pipe(fds);

}
