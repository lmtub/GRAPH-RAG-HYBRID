void qemu_chr_be_write(CharDriverState *s, uint8_t *buf, int len)

{

    s->chr_read(s->handler_opaque, buf, len);

}
