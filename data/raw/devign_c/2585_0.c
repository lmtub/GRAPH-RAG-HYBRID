static int client_migrate_info(Monitor *mon, const QDict *qdict,

                               QObject **ret_data)

{

    const char *protocol = qdict_get_str(qdict, "protocol");

    const char *hostname = qdict_get_str(qdict, "hostname");

    const char *subject  = qdict_get_try_str(qdict, "cert-subject");

    int port             = qdict_get_try_int(qdict, "port", -1);

    int tls_port         = qdict_get_try_int(qdict, "tls-port", -1);

    Error *err = NULL;

    int ret;



    if (strcmp(protocol, "spice") == 0) {

        if (!qemu_using_spice(&err)) {

            qerror_report_err(err);

            error_free(err);

            return -1;

        }



        if (port == -1 && tls_port == -1) {

            qerror_report(QERR_MISSING_PARAMETER, "port/tls-port");

            return -1;

        }



        ret = qemu_spice_migrate_info(hostname, port, tls_port, subject);

        if (ret != 0) {

            qerror_report(QERR_UNDEFINED_ERROR);

            return -1;

        }

        return 0;

    }



    qerror_report(QERR_INVALID_PARAMETER, "protocol");

    return -1;

}
