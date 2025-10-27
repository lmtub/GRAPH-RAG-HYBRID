static CharDriverState *qmp_chardev_open_parallel(const char *id,

                                                  ChardevBackend *backend,

                                                  ChardevReturn *ret,

                                                  Error **errp)

{

    ChardevHostdev *parallel = backend->u.parallel;

    int fd;



    fd = qmp_chardev_open_file_source(parallel->device, O_RDWR, errp);

    if (fd < 0) {

        return NULL;

    }

    return qemu_chr_open_pp_fd(fd, errp);

}
