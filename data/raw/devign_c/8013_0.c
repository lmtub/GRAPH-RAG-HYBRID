void qemu_set_cloexec(int fd)

{

    int f;

    f = fcntl(fd, F_GETFD);

    fcntl(fd, F_SETFD, f | FD_CLOEXEC);

}
