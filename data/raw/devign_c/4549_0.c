static void readline_printf_func(void *opaque, const char *fmt, ...)

{

    va_list ap;

    va_start(ap, fmt);

    vprintf(fmt, ap);

    va_end(ap);

}
