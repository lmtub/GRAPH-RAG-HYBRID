void qerror_print(QError *qerror)

{

    QString *qstring = qerror_human(qerror);

    loc_push_restore(&qerror->loc);

    error_report("%s", qstring_get_str(qstring));

    loc_pop(&qerror->loc);

    QDECREF(qstring);

}
