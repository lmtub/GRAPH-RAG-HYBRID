void qemu_chr_printf(CharDriverState *s, const char *fmt, ...)

{

    char buf[4096];

    va_list ap;

    va_start(ap, fmt);

    vsnprintf(buf, sizeof(buf), fmt, ap);

    qemu_chr_write(s, (uint8_t *)buf, strlen(buf));

    va_end(ap);

}
