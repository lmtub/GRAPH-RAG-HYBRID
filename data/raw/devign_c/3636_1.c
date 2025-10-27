static void serial_receive1(void *opaque, const uint8_t *buf, int size)

{

    SerialState *s = opaque;

    serial_receive_byte(s, buf[0]);

}
