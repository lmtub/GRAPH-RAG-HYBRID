void qemu_error_internal(const char *file, int linenr, const char *func,

                         const char *fmt, ...)

{

    va_list va;

    QError *qerror;



    assert(qemu_error_sink != NULL);



    va_start(va, fmt);

    qerror = qerror_from_info(file, linenr, func, fmt, &va);

    va_end(va);



    switch (qemu_error_sink->dest) {

    case ERR_SINK_FILE:

        qerror_print(qerror);

        QDECREF(qerror);

        break;

    case ERR_SINK_MONITOR:

        assert(qemu_error_sink->mon->error == NULL);

        qemu_error_sink->mon->error = qerror;

        break;

    }

}
