static CharDriverState *qmp_chardev_open_serial(const char *id,

                                                ChardevBackend *backend,

                                                ChardevReturn *ret,

                                                Error **errp)

{

    ChardevHostdev *serial = backend->serial;

    int fd;



    fd = qmp_chardev_open_file_source(serial->device, O_RDWR, errp);

    if (fd < 0) {

        return NULL;

    }

    qemu_set_nonblock(fd);

    return qemu_chr_open_tty_fd(fd);

}
