int avio_printf(AVIOContext *s, const char *fmt, ...)

{

    va_list ap;

    char buf[4096];

    int ret;



    va_start(ap, fmt);

    ret = vsnprintf(buf, sizeof(buf), fmt, ap);

    va_end(ap);

    avio_write(s, buf, strlen(buf));

    return ret;

}
