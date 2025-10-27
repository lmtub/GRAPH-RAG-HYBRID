static int protocol_client_vencrypt_auth(VncState *vs, uint8_t *data, size_t len)

{

    int auth = read_u32(data, 0);



    if (auth != vs->vd->subauth) {

	VNC_DEBUG("Rejecting auth %d\n", auth);

	vnc_write_u8(vs, 0); /* Reject auth */

	vnc_flush(vs);

	vnc_client_error(vs);

    } else {

	VNC_DEBUG("Accepting auth %d, starting handshake\n", auth);

	vnc_write_u8(vs, 1); /* Accept auth */

	vnc_flush(vs);



	if (vnc_start_tls(vs) < 0) {

	    VNC_DEBUG("Failed to complete TLS\n");

	    return 0;

	}



	if (vs->wiremode == VNC_WIREMODE_TLS) {

	    VNC_DEBUG("Starting VeNCrypt subauth\n");

	    return start_auth_vencrypt_subauth(vs);

	} else {

	    VNC_DEBUG("TLS handshake blocked\n");

	    return 0;

	}

    }

    return 0;

}
