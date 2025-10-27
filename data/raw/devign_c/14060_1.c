static int read_random(uint32_t *dst, const char *file)

{

    int fd = open(file, O_RDONLY);

    int err = -1;



    if (fd == -1)

        return -1;

#if HAVE_FCNTL && defined(O_NONBLOCK)

    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) != -1)

#endif

        err = read(fd, dst, sizeof(*dst));

    close(fd);



    return err;

}
