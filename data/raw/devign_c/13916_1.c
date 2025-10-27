static CharDriverState *qemu_chr_open_tty_fd(int fd,

                                             ChardevCommon *backend,

                                             Error **errp)

{

    CharDriverState *chr;



    tty_serial_init(fd, 115200, 'N', 8, 1);

    chr = qemu_chr_open_fd(fd, fd, backend, errp);




    chr->chr_ioctl = tty_serial_ioctl;

    chr->chr_close = qemu_chr_close_tty;

    return chr;
