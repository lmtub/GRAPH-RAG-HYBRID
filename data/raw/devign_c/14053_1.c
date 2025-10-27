QemuOpts *qemu_opts_find(QemuOptsList *list, const char *id)

{

    QemuOpts *opts;



    QTAILQ_FOREACH(opts, &list->head, next) {

        if (!opts->id) {

            if (!id) {

                return opts;

            }

            continue;

        }

        if (strcmp(opts->id, id) != 0) {

            continue;

        }

        return opts;

    }

    return NULL;

}
