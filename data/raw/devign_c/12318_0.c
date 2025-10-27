static void rtl8139_transfer_frame(RTL8139State *s, const uint8_t *buf, int size, int do_interrupt)

{

    if (!size)

    {

        DEBUG_PRINT(("RTL8139: +++ empty ethernet frame\n"));

        return;

    }



    if (TxLoopBack == (s->TxConfig & TxLoopBack))

    {

        DEBUG_PRINT(("RTL8139: +++ transmit loopback mode\n"));

        rtl8139_do_receive(s, buf, size, do_interrupt);

    }

    else

    {

        qemu_send_packet(s->vc, buf, size);

    }

}
