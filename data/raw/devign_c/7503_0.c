void hmp_info_vnc(Monitor *mon, const QDict *qdict)

{

    VncInfo *info;

    Error *err = NULL;

    VncClientInfoList *client;



    info = qmp_query_vnc(&err);

    if (err) {

        monitor_printf(mon, "%s\n", error_get_pretty(err));

        error_free(err);

        return;

    }



    if (!info->enabled) {

        monitor_printf(mon, "Server: disabled\n");

        goto out;

    }



    monitor_printf(mon, "Server:\n");

    if (info->has_host && info->has_service) {

        monitor_printf(mon, "     address: %s:%s\n", info->host, info->service);

    }

    if (info->has_auth) {

        monitor_printf(mon, "        auth: %s\n", info->auth);

    }



    if (!info->has_clients || info->clients == NULL) {

        monitor_printf(mon, "Client: none\n");

    } else {

        for (client = info->clients; client; client = client->next) {

            monitor_printf(mon, "Client:\n");

            monitor_printf(mon, "     address: %s:%s\n",

                           client->value->base->host,

                           client->value->base->service);

            monitor_printf(mon, "  x509_dname: %s\n",

                           client->value->x509_dname ?

                           client->value->x509_dname : "none");

            monitor_printf(mon, "    username: %s\n",

                           client->value->has_sasl_username ?

                           client->value->sasl_username : "none");

        }

    }



out:

    qapi_free_VncInfo(info);

}
