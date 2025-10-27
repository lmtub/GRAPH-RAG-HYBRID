void vncws_tls_handshake_io(void *opaque)

{

    VncState *vs = (VncState *)opaque;



    if (!vs->tls.session) {

        VNC_DEBUG("TLS Websocket setup\n");

        if (vnc_tls_client_setup(vs, vs->vd->tls.x509cert != NULL) < 0) {

            return;

        }

    }

    VNC_DEBUG("Handshake IO continue\n");

    vncws_start_tls_handshake(vs);

}
