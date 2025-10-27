static void monitor_readline_printf(void *opaque, const char *fmt, ...)

{

    va_list ap;

    va_start(ap, fmt);

    monitor_vprintf(opaque, fmt, ap);

    va_end(ap);

}
