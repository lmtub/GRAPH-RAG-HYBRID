static void parse_chap(struct iscsi_context *iscsi, const char *target,

                       Error **errp)

{

    QemuOptsList *list;

    QemuOpts *opts;

    const char *user = NULL;

    const char *password = NULL;

    const char *secretid;

    char *secret = NULL;



    list = qemu_find_opts("iscsi");

    if (!list) {

        return;

    }



    opts = qemu_opts_find(list, target);

    if (opts == NULL) {

        opts = QTAILQ_FIRST(&list->head);

        if (!opts) {

            return;

        }

    }



    user = qemu_opt_get(opts, "user");

    if (!user) {

        return;

    }



    secretid = qemu_opt_get(opts, "password-secret");

    password = qemu_opt_get(opts, "password");

    if (secretid && password) {

        error_setg(errp, "'password' and 'password-secret' properties are "

                   "mutually exclusive");

        return;

    }

    if (secretid) {

        secret = qcrypto_secret_lookup_as_utf8(secretid, errp);

        if (!secret) {

            return;

        }

        password = secret;

    } else if (!password) {

        error_setg(errp, "CHAP username specified but no password was given");

        return;

    }



    if (iscsi_set_initiator_username_pwd(iscsi, user, password)) {

        error_setg(errp, "Failed to set initiator username and password");

    }



    g_free(secret);

}
