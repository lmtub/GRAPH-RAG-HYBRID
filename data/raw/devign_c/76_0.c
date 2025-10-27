static void uart_rx_reset(UartState *s)

{

    s->rx_wpos = 0;

    s->rx_count = 0;

    qemu_chr_accept_input(s->chr);



    s->r[R_SR] |= UART_SR_INTR_REMPTY;

    s->r[R_SR] &= ~UART_SR_INTR_RFUL;

}
