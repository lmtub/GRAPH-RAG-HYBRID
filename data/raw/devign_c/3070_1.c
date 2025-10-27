static void serial_tx_done(void *opaque)

{

    SerialState *s = opaque;



    if (s->tx_burst < 0) {

        uint16_t divider;



        if (s->divider)

          divider = s->divider;

        else

          divider = 1;



        /* We assume 10 bits/char, OK for this purpose. */

        s->tx_burst = THROTTLE_TX_INTERVAL * 1000 /

            (1000000 * 10 / (s->baudbase / divider));

    }

    s->thr_ipending = 1;

    s->lsr |= UART_LSR_THRE;

    s->lsr |= UART_LSR_TEMT;

    serial_update_irq(s);

}
