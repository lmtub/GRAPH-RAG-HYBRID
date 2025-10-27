static void vnc_qmp_event(VncState *vs, QAPIEvent event)

{

    VncServerInfo *si;



    if (!vs->info) {

        return;

    }

    g_assert(vs->info->base);



    si = vnc_server_info_get(vs->vd);

    if (!si) {

        return;

    }



    switch (event) {

    case QAPI_EVENT_VNC_CONNECTED:

        qapi_event_send_vnc_connected(si, vs->info->base, &error_abort);

        break;

    case QAPI_EVENT_VNC_INITIALIZED:

        qapi_event_send_vnc_initialized(si, vs->info, &error_abort);

        break;

    case QAPI_EVENT_VNC_DISCONNECTED:

        qapi_event_send_vnc_disconnected(si, vs->info, &error_abort);

        break;

    default:

        break;

    }



    qapi_free_VncServerInfo(si);

}
