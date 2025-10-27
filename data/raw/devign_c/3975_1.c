void error_set_win32(Error **errp, int win32_err, ErrorClass err_class,

                     const char *fmt, ...)

{

    va_list ap;

    char *msg1, *msg2;



    if (errp == NULL) {

        return;

    }



    va_start(ap, fmt);

    error_setv(errp, err_class, fmt, ap);

    va_end(ap);



    if (win32_err != 0) {

        msg1 = (*errp)->msg;

        msg2 = g_win32_error_message(win32_err);

        (*errp)->msg = g_strdup_printf("%s: %s (error: %x)", msg1, msg2,

                                       (unsigned)win32_err);

        g_free(msg2);

        g_free(msg1);

    }

}
