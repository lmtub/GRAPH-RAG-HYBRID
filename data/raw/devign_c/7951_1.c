int ff_socket(int af, int type, int proto)

{

    int fd;



#ifdef SOCK_CLOEXEC

    fd = socket(af, type | SOCK_CLOEXEC, proto);

    if (fd == -1 && errno == EINVAL)

#endif

    {

        fd = socket(af, type, proto);

#if HAVE_FCNTL

        if (fd != -1)

            fcntl(fd, F_SETFD, FD_CLOEXEC);

#endif

    }

    return fd;

}
