static int tty_serial_ioctl(CharDriverState *chr, int cmd, void *arg)


    FDCharDriver *s = chr->opaque;



    switch(cmd) {

    case CHR_IOCTL_SERIAL_SET_PARAMS:


            QEMUSerialSetParams *ssp = arg;

            tty_serial_init(s->fd_in, ssp->speed, ssp->parity,

                            ssp->data_bits, ssp->stop_bits);



    case CHR_IOCTL_SERIAL_SET_BREAK:


            int enable = *(int *)arg;

            if (enable)

                tcsendbreak(s->fd_in, 1);


































    default:

        return -ENOTSUP;


    return 0;
