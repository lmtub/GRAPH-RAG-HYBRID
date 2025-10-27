QemuOpts *qemu_opts_create(QemuOptsList *list, const char *id,

                           int fail_if_exists, Error **errp)

{

    QemuOpts *opts = NULL;



    if (id) {

        if (!id_wellformed(id)) {

            error_set(errp,QERR_INVALID_PARAMETER_VALUE, "id", "an identifier");

#if 0 /* conversion from qerror_report() to error_set() broke this: */

            error_printf_unless_qmp("Identifiers consist of letters, digits, '-', '.', '_', starting with a letter.\n");

#endif

            return NULL;

        }

        opts = qemu_opts_find(list, id);

        if (opts != NULL) {

            if (fail_if_exists && !list->merge_lists) {

                error_setg(errp, "Duplicate ID '%s' for %s", id, list->name);

                return NULL;

            } else {

                return opts;

            }

        }

    } else if (list->merge_lists) {

        opts = qemu_opts_find(list, NULL);

        if (opts) {

            return opts;

        }

    }

    opts = g_malloc0(sizeof(*opts));

    opts->id = g_strdup(id);

    opts->list = list;

    loc_save(&opts->loc);

    QTAILQ_INIT(&opts->head);

    QTAILQ_INSERT_TAIL(&list->head, opts, next);

    return opts;

}
