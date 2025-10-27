static void rxfilter_notify(NetClientState *nc)

{

    QObject *event_data;

    VirtIONet *n = qemu_get_nic_opaque(nc);



    if (nc->rxfilter_notify_enabled) {

        if (n->netclient_name) {

            event_data = qobject_from_jsonf("{ 'name': %s, 'path': %s }",

                                    n->netclient_name,

                                    object_get_canonical_path(OBJECT(n->qdev)));

        } else {

            event_data = qobject_from_jsonf("{ 'path': %s }",

                                    object_get_canonical_path(OBJECT(n->qdev)));

        }

        monitor_protocol_event(QEVENT_NIC_RX_FILTER_CHANGED, event_data);

        qobject_decref(event_data);



        /* disable event notification to avoid events flooding */

        nc->rxfilter_notify_enabled = 0;

    }

}
