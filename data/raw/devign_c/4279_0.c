static int vnc_start_tls(struct VncState *vs) {

    static const int cert_type_priority[] = { GNUTLS_CRT_X509, 0 };

    static const int protocol_priority[]= { GNUTLS_TLS1_1, GNUTLS_TLS1_0, GNUTLS_SSL3, 0 };

    static const int kx_anon[] = {GNUTLS_KX_ANON_DH, 0};

    static const int kx_x509[] = {GNUTLS_KX_DHE_DSS, GNUTLS_KX_RSA, GNUTLS_KX_DHE_RSA, GNUTLS_KX_SRP, 0};



    VNC_DEBUG("Do TLS setup\n");

    if (vnc_tls_initialize() < 0) {

	VNC_DEBUG("Failed to init TLS\n");

	vnc_client_error(vs);

	return -1;

    }

    if (vs->tls_session == NULL) {

	if (gnutls_init(&vs->tls_session, GNUTLS_SERVER) < 0) {

	    vnc_client_error(vs);

	    return -1;

	}



	if (gnutls_set_default_priority(vs->tls_session) < 0) {

	    gnutls_deinit(vs->tls_session);

	    vs->tls_session = NULL;

	    vnc_client_error(vs);

	    return -1;

	}



	if (gnutls_kx_set_priority(vs->tls_session, NEED_X509_AUTH(vs) ? kx_x509 : kx_anon) < 0) {

	    gnutls_deinit(vs->tls_session);

	    vs->tls_session = NULL;

	    vnc_client_error(vs);

	    return -1;

	}



	if (gnutls_certificate_type_set_priority(vs->tls_session, cert_type_priority) < 0) {

	    gnutls_deinit(vs->tls_session);

	    vs->tls_session = NULL;

	    vnc_client_error(vs);

	    return -1;

	}



	if (gnutls_protocol_set_priority(vs->tls_session, protocol_priority) < 0) {

	    gnutls_deinit(vs->tls_session);

	    vs->tls_session = NULL;

	    vnc_client_error(vs);

	    return -1;

	}



	if (NEED_X509_AUTH(vs)) {

	    gnutls_certificate_server_credentials x509_cred = vnc_tls_initialize_x509_cred(vs);

	    if (!x509_cred) {

		gnutls_deinit(vs->tls_session);

		vs->tls_session = NULL;

		vnc_client_error(vs);

		return -1;

	    }

	    if (gnutls_credentials_set(vs->tls_session, GNUTLS_CRD_CERTIFICATE, x509_cred) < 0) {

		gnutls_deinit(vs->tls_session);

		vs->tls_session = NULL;

		gnutls_certificate_free_credentials(x509_cred);

		vnc_client_error(vs);

		return -1;

	    }

	    if (vs->vd->x509verify) {

		VNC_DEBUG("Requesting a client certificate\n");

		gnutls_certificate_server_set_request (vs->tls_session, GNUTLS_CERT_REQUEST);

	    }



	} else {

	    gnutls_anon_server_credentials anon_cred = vnc_tls_initialize_anon_cred();

	    if (!anon_cred) {

		gnutls_deinit(vs->tls_session);

		vs->tls_session = NULL;

		vnc_client_error(vs);

		return -1;

	    }

	    if (gnutls_credentials_set(vs->tls_session, GNUTLS_CRD_ANON, anon_cred) < 0) {

		gnutls_deinit(vs->tls_session);

		vs->tls_session = NULL;

		gnutls_anon_free_server_credentials(anon_cred);

		vnc_client_error(vs);

		return -1;

	    }

	}



	gnutls_transport_set_ptr(vs->tls_session, (gnutls_transport_ptr_t)vs);

	gnutls_transport_set_push_function(vs->tls_session, vnc_tls_push);

	gnutls_transport_set_pull_function(vs->tls_session, vnc_tls_pull);

    }



    VNC_DEBUG("Start TLS handshake process\n");

    return vnc_continue_handshake(vs);

}
