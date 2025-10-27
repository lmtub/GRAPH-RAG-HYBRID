void vncws_tls_handshake_peek(void *opaque)

{

    VncState *vs = opaque;

    long ret;



    if (!vs->ws_tls.session) {

        char peek[4];

        ret = qemu_recv(vs->csock, peek, sizeof(peek), MSG_PEEK);

        if (ret && (strncmp(peek, "\x16", 1) == 0

                    || strncmp(peek, "\x80", 1) == 0)) {

            VNC_DEBUG("TLS Websocket connection recognized");

            vnc_tls_client_setup(vs, 1);

            vncws_start_tls_handshake(vs);

        } else {

            vncws_handshake_read(vs);

        }

    } else {

        qemu_set_fd_handler2(vs->csock, NULL, vncws_handshake_read, NULL, vs);

    }

}
