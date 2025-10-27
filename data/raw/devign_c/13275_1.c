static void error_setv(Error **errp, ErrorClass err_class,

                       const char *fmt, va_list ap)

{

    Error *err;

    int saved_errno = errno;



    if (errp == NULL) {

        return;

    }

    assert(*errp == NULL);



    err = g_malloc0(sizeof(*err));

    err->msg = g_strdup_vprintf(fmt, ap);

    err->err_class = err_class;



    if (errp == &error_abort) {

        error_report_err(err);

        abort();

    }



    *errp = err;



    errno = saved_errno;

}
