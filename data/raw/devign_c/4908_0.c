VncInfo2List *qmp_query_vnc_servers(Error **errp)

{

    VncInfo2List *item, *prev = NULL;

    VncInfo2 *info;

    VncDisplay *vd;

    DeviceState *dev;



    QTAILQ_FOREACH(vd, &vnc_displays, next) {

        info = g_new0(VncInfo2, 1);

        info->id = g_strdup(vd->id);

        info->clients = qmp_query_client_list(vd);

        qmp_query_auth(vd, info);

        if (vd->dcl.con) {

            dev = DEVICE(object_property_get_link(OBJECT(vd->dcl.con),

                                                  "device", NULL));

            info->has_display = true;

            info->display = g_strdup(dev->id);

        }

        if (vd->lsock != NULL) {

            info->server = qmp_query_server_entry(

                vd->lsock, false, info->server);

        }

        if (vd->lwebsock != NULL) {

            info->server = qmp_query_server_entry(

                vd->lwebsock, true, info->server);

        }



        item = g_new0(VncInfo2List, 1);

        item->value = info;

        item->next = prev;

        prev = item;

    }

    return prev;

}
