static int monitor_fprintf(FILE *stream, const char *fmt, ...)

{

    va_list ap;

    va_start(ap, fmt);

    monitor_vprintf((Monitor *)stream, fmt, ap);

    va_end(ap);

    return 0;

}
