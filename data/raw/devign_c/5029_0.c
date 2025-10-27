static int nbd_config(BDRVNBDState *s, QDict *options, char **export)

{

    Error *local_err = NULL;



    if (qdict_haskey(options, "path") == qdict_haskey(options, "host")) {

        if (qdict_haskey(options, "path")) {

            qerror_report(ERROR_CLASS_GENERIC_ERROR, "path and host may not "

                          "be used at the same time.");

        } else {

            qerror_report(ERROR_CLASS_GENERIC_ERROR, "one of path and host "

                          "must be specified.");

        }

        return -EINVAL;

    }



    s->client.is_unix = qdict_haskey(options, "path");

    s->socket_opts = qemu_opts_create(&socket_optslist, NULL, 0,

                                      &error_abort);



    qemu_opts_absorb_qdict(s->socket_opts, options, &local_err);

    if (local_err) {

        qerror_report_err(local_err);

        error_free(local_err);

        return -EINVAL;

    }



    if (!qemu_opt_get(s->socket_opts, "port")) {

        qemu_opt_set_number(s->socket_opts, "port", NBD_DEFAULT_PORT);

    }



    *export = g_strdup(qdict_get_try_str(options, "export"));

    if (*export) {

        qdict_del(options, "export");

    }



    return 0;

}
