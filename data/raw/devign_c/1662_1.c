int qemu_create_pidfile(const char *filename)

{

    char buffer[128];

    int len;

    int fd;



    fd = qemu_open(filename, O_RDWR | O_CREAT, 0600);

    if (fd == -1) {

        return -1;

    }

    if (lockf(fd, F_TLOCK, 0) == -1) {

        close(fd);

        return -1;

    }

    len = snprintf(buffer, sizeof(buffer), FMT_pid "\n", getpid());

    if (write(fd, buffer, len) != len) {

        close(fd);

        return -1;

    }



    close(fd);

    return 0;

}
