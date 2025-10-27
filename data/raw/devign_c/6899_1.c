static int serial_can_receive(SerialState *s)

{

    return !(s->lsr & UART_LSR_DR);

}
