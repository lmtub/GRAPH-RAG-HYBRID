void error_setg_errno(Error **errp, int os_errno, const char *fmt, ...)

{

    va_list ap;

    char *msg;

    int saved_errno = errno;



    if (errp == NULL) {

        return;

    }



    va_start(ap, fmt);

    error_setv(errp, ERROR_CLASS_GENERIC_ERROR, fmt, ap);

    va_end(ap);



    if (os_errno != 0) {

        msg = (*errp)->msg;

        (*errp)->msg = g_strdup_printf("%s: %s", msg, strerror(os_errno));

        g_free(msg);

    }



    errno = saved_errno;

}
