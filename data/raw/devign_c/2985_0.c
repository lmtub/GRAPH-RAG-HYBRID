static void term_read(void *opaque, const uint8_t *buf, int size)

{

    int i;

    for(i = 0; i < size; i++)

        term_handle_byte(buf[i]);

}
