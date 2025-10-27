static void rtl8139_receive(void *opaque, const uint8_t *buf, size_t size)

{

    rtl8139_do_receive(opaque, buf, size, 1);

}
