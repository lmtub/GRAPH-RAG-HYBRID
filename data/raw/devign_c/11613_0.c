static void qerror_abort(const QError *qerr, const char *fmt, ...)

{

    va_list ap;



    fprintf(stderr, "qerror: bad call in function '%s':\n", qerr->func);

    fprintf(stderr, "qerror: -> ");



    va_start(ap, fmt);

    vfprintf(stderr, fmt, ap);

    va_end(ap);



    fprintf(stderr, "\nqerror: call at %s:%d\n", qerr->file, qerr->linenr);

    abort();

}
