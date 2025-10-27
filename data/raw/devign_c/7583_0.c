static int protocol_version(VncState *vs, uint8_t *version, size_t len)

{

    char local[13];



    memcpy(local, version, 12);

    local[12] = 0;



    if (sscanf(local, "RFB %03d.%03d\n", &vs->major, &vs->minor) != 2) {

	VNC_DEBUG("Malformed protocol version %s\n", local);

	vnc_client_error(vs);

	return 0;

    }

    VNC_DEBUG("Client request protocol version %d.%d\n", vs->major, vs->minor);

    if (vs->major != 3 ||

	(vs->minor != 3 &&

	 vs->minor != 4 &&

	 vs->minor != 5 &&

	 vs->minor != 7 &&

	 vs->minor != 8)) {

	VNC_DEBUG("Unsupported client version\n");

	vnc_write_u32(vs, VNC_AUTH_INVALID);

	vnc_flush(vs);

	vnc_client_error(vs);

	return 0;

    }

    /* Some broken clients report v3.4 or v3.5, which spec requires to be treated

     * as equivalent to v3.3 by servers

     */

    if (vs->minor == 4 || vs->minor == 5)

	vs->minor = 3;



    if (vs->minor == 3) {

	if (vs->vd->auth == VNC_AUTH_NONE) {

            VNC_DEBUG("Tell client auth none\n");

            vnc_write_u32(vs, vs->vd->auth);

            vnc_flush(vs);

            vnc_read_when(vs, protocol_client_init, 1);

       } else if (vs->vd->auth == VNC_AUTH_VNC) {

            VNC_DEBUG("Tell client VNC auth\n");

            vnc_write_u32(vs, vs->vd->auth);

            vnc_flush(vs);

            start_auth_vnc(vs);

       } else {

            VNC_DEBUG("Unsupported auth %d for protocol 3.3\n", vs->vd->auth);

            vnc_write_u32(vs, VNC_AUTH_INVALID);

            vnc_flush(vs);

            vnc_client_error(vs);

       }

    } else {

	VNC_DEBUG("Telling client we support auth %d\n", vs->vd->auth);

	vnc_write_u8(vs, 1); /* num auth */

	vnc_write_u8(vs, vs->vd->auth);

	vnc_read_when(vs, protocol_client_auth, 1);

	vnc_flush(vs);

    }



    return 0;

}
