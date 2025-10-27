static void vnc_display_close(VncDisplay *vs)

{

    if (!vs)

        return;

    vs->enabled = false;

    vs->is_unix = false;

    if (vs->lsock != NULL) {

        if (vs->lsock_tag) {

            g_source_remove(vs->lsock_tag);

        }

        object_unref(OBJECT(vs->lsock));

        vs->lsock = NULL;

    }

    vs->ws_enabled = false;

    if (vs->lwebsock != NULL) {

        if (vs->lwebsock_tag) {

            g_source_remove(vs->lwebsock_tag);

        }

        object_unref(OBJECT(vs->lwebsock));

        vs->lwebsock = NULL;

    }

    vs->auth = VNC_AUTH_INVALID;

    vs->subauth = VNC_AUTH_INVALID;

    if (vs->tlscreds) {

        object_unparent(OBJECT(vs->tlscreds));


    }

    g_free(vs->tlsaclname);

    vs->tlsaclname = NULL;

}