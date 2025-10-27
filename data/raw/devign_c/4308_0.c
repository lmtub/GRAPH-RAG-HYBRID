static void uart_send_breaks(UartState *s)

{

    int break_enabled = 1;



    qemu_chr_fe_ioctl(s->chr, CHR_IOCTL_SERIAL_SET_BREAK,

                               &break_enabled);

}
