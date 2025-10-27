static void vnc_display_close(VncDisplay *vd)

{

    size_t i;

    if (!vd) {

        return;

    }

    vd->is_unix = false;

    for (i = 0; i < vd->nlsock; i++) {

        if (vd->lsock_tag[i]) {

            g_source_remove(vd->lsock_tag[i]);

        }

        object_unref(OBJECT(vd->lsock[i]));

    }

    g_free(vd->lsock);

    g_free(vd->lsock_tag);

    vd->lsock = NULL;

    vd->lsock_tag = NULL;

    vd->nlsock = 0;



    for (i = 0; i < vd->nlwebsock; i++) {

        if (vd->lwebsock_tag[i]) {

            g_source_remove(vd->lwebsock_tag[i]);

        }

        object_unref(OBJECT(vd->lwebsock[i]));

    }

    g_free(vd->lwebsock);

    g_free(vd->lwebsock_tag);

    vd->lwebsock = NULL;

    vd->lwebsock_tag = NULL;

    vd->nlwebsock = 0;



    vd->auth = VNC_AUTH_INVALID;

    vd->subauth = VNC_AUTH_INVALID;

    if (vd->tlscreds) {

        object_unparent(OBJECT(vd->tlscreds));

        vd->tlscreds = NULL;

    }

    g_free(vd->tlsaclname);

    vd->tlsaclname = NULL;

    if (vd->lock_key_sync) {

        qemu_remove_led_event_handler(vd->led);


    }

}