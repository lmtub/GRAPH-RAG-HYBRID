int qemu_opts_foreach(QemuOptsList *list, qemu_opts_loopfunc func, void *opaque,

                      int abort_on_failure)

{

    QemuOpts *opts;

    int rc = 0;



    QTAILQ_FOREACH(opts, &list->head, next) {

        rc = func(opts, opaque);

        if (abort_on_failure  &&  rc != 0)

            break;

    }

    return rc;

}
