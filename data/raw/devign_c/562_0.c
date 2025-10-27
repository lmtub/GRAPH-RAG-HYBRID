static bool nbd_process_legacy_socket_options(QDict *output_options,

                                              QemuOpts *legacy_opts,

                                              Error **errp)

{

    const char *path = qemu_opt_get(legacy_opts, "path");

    const char *host = qemu_opt_get(legacy_opts, "host");

    const char *port = qemu_opt_get(legacy_opts, "port");

    const QDictEntry *e;



    if (!path && !host && !port) {

        return true;

    }



    for (e = qdict_first(output_options); e; e = qdict_next(output_options, e))

    {

        if (strstart(e->key, "server.", NULL)) {

            error_setg(errp, "Cannot use 'server' and path/host/port at the "

                       "same time");

            return false;

        }

    }



    if (path && host) {

        error_setg(errp, "path and host may not be used at the same time");

        return false;

    } else if (path) {

        if (port) {

            error_setg(errp, "port may not be used without host");

            return false;

        }



        qdict_put(output_options, "server.type", qstring_from_str("unix"));

        qdict_put(output_options, "server.data.path", qstring_from_str(path));

    } else if (host) {

        qdict_put(output_options, "server.type", qstring_from_str("inet"));

        qdict_put(output_options, "server.data.host", qstring_from_str(host));

        qdict_put(output_options, "server.data.port",

                  qstring_from_str(port ?: stringify(NBD_DEFAULT_PORT)));

    }



    return true;

}
