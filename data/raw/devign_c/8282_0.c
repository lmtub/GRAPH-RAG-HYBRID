QError *qerror_from_info(const char *file, int linenr, const char *func,

                         const char *fmt, va_list *va)

{

    QError *qerr;



    qerr = qerror_new();

    loc_save(&qerr->loc);

    qerr->linenr = linenr;

    qerr->file = file;

    qerr->func = func;



    if (!fmt) {

        qerror_abort(qerr, "QDict not specified");

    }



    qerror_set_data(qerr, fmt, va);

    qerror_set_desc(qerr, fmt);



    return qerr;

}
