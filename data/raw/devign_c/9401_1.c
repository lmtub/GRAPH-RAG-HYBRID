static int parse_chap(struct iscsi_context *iscsi, const char *target)

{

    QemuOptsList *list;

    QemuOpts *opts;

    const char *user = NULL;

    const char *password = NULL;



    list = qemu_find_opts("iscsi");

    if (!list) {

        return 0;

    }



    opts = qemu_opts_find(list, target);

    if (opts == NULL) {

        opts = QTAILQ_FIRST(&list->head);

        if (!opts) {

            return 0;

        }

    }



    user = qemu_opt_get(opts, "user");

    if (!user) {

        return 0;

    }



    password = qemu_opt_get(opts, "password");

    if (!password) {

        error_report("CHAP username specified but no password was given");

        return -1;

    }



    if (iscsi_set_initiator_username_pwd(iscsi, user, password)) {

        error_report("Failed to set initiator username and password");

        return -1;

    }



    return 0;

}
