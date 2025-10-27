QemuOpts *qemu_opts_create(QemuOptsList *list, const char *id, int fail_if_exists)

{

    QemuOpts *opts = NULL;



    if (id) {

        if (!id_wellformed(id)) {

            qerror_report(QERR_INVALID_PARAMETER_VALUE, "id", "an identifier");

            error_printf_unless_qmp("Identifiers consist of letters, digits, '-', '.', '_', starting with a letter.\n");

            return NULL;

        }

        opts = qemu_opts_find(list, id);

        if (opts != NULL) {

            if (fail_if_exists && !list->merge_lists) {

                qerror_report(QERR_DUPLICATE_ID, id, list->name);

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

    if (id) {

        opts->id = g_strdup(id);

    }

    opts->list = list;

    loc_save(&opts->loc);

    QTAILQ_INIT(&opts->head);

    QTAILQ_INSERT_TAIL(&list->head, opts, next);

    return opts;

}
