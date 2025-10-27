static void serial_reset(void *opaque)

{

    SerialState *s = opaque;



    s->divider = 0;

    s->rbr = 0;

    s->ier = 0;

    s->iir = UART_IIR_NO_INT;

    s->lcr = 0;

    s->mcr = 0;

    s->lsr = UART_LSR_TEMT | UART_LSR_THRE;

    s->msr = UART_MSR_DCD | UART_MSR_DSR | UART_MSR_CTS;

    s->scr = 0;



    s->thr_ipending = 0;

    s->last_break_enable = 0;

    qemu_irq_lower(s->irq);

}
