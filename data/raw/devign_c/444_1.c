static int floppy_probe_device(const char *filename)

{

    int fd, ret;

    int prio = 0;

    struct floppy_struct fdparam;

    struct stat st;



    if (strstart(filename, "/dev/fd", NULL) &&

        !strstart(filename, "/dev/fdset/", NULL)) {

        prio = 50;

    }



    fd = qemu_open(filename, O_RDONLY | O_NONBLOCK);

    if (fd < 0) {

        goto out;

    }

    ret = fstat(fd, &st);

    if (ret == -1 || !S_ISBLK(st.st_mode)) {

        goto outc;

    }



    /* Attempt to detect via a floppy specific ioctl */

    ret = ioctl(fd, FDGETPRM, &fdparam);

    if (ret >= 0)

        prio = 100;



outc:

    qemu_close(fd);

out:

    return prio;

}
