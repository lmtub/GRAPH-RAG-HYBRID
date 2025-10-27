static CharDriverState *qmp_chardev_open_parallel(ChardevHostdev *parallel,

                                                  Error **errp)

{

#ifdef HAVE_CHARDEV_PARPORT

    int fd;



    fd = qmp_chardev_open_file_source(parallel->device, O_RDWR, errp);

    if (error_is_set(errp)) {

        return NULL;

    }

    return qemu_chr_open_pp_fd(fd);

#else

    error_setg(errp, "character device backend type 'parallel' not supported");

    return NULL;

#endif

}
