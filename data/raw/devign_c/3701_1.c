int qemu_opts_foreach(QemuOptsList *list, qemu_opts_loopfunc func,

                      void *opaque, Error **errp)

{

    Location loc;

    QemuOpts *opts;

    int rc;



    loc_push_none(&loc);

    QTAILQ_FOREACH(opts, &list->head, next) {

        loc_restore(&opts->loc);

        rc = func(opaque, opts, errp);

        if (rc) {

            return rc;

        }

        assert(!errp || !*errp);

    }

    loc_pop(&loc);

    return 0;

}
