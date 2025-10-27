dprint(int level, const char *fmt, ...)

{

    va_list args;



    if (level <= debug) {

        va_start(args, fmt);

        vfprintf(stderr, fmt, args);

        va_end(args);

    }

}
