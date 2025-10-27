void vnc_tls_client_cleanup(VncState *vs)

{

    if (vs->tls.session) {

        gnutls_deinit(vs->tls.session);

        vs->tls.session = NULL;

    }

    g_free(vs->tls.dname);

}
