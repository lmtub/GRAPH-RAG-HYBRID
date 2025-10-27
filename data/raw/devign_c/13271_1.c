static CharDriverState *qemu_chr_open_pp_fd(int fd,

                                            ChardevCommon *backend,

                                            Error **errp)

{

    CharDriverState *chr;

    ParallelCharDriver *drv;



    if (ioctl(fd, PPCLAIM) < 0) {

        error_setg_errno(errp, errno, "not a parallel port");

        close(fd);

        return NULL;

    }



    drv = g_new0(ParallelCharDriver, 1);

    drv->fd = fd;

    drv->mode = IEEE1284_MODE_COMPAT;



    chr = qemu_chr_alloc(backend, errp);

    if (!chr) {

        return NULL;

    }

    chr->chr_write = null_chr_write;

    chr->chr_ioctl = pp_ioctl;

    chr->chr_close = pp_close;

    chr->opaque = drv;



    return chr;

}
