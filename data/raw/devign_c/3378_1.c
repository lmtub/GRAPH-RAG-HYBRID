static int protocol_client_vencrypt_init(VncState *vs, uint8_t *data, size_t len)

{

    if (data[0] != 0 ||

        data[1] != 2) {

        VNC_DEBUG("Unsupported VeNCrypt protocol %d.%d\n", (int)data[0], (int)data[1]);

        vnc_write_u8(vs, 1); /* Reject version */

        vnc_flush(vs);

        vnc_client_error(vs);

    } else {

        VNC_DEBUG("Sending allowed auth %d\n", vs->subauth);

        vnc_write_u8(vs, 0); /* Accept version */

        vnc_write_u8(vs, 1); /* Number of sub-auths */

        vnc_write_u32(vs, vs->subauth); /* The supported auth */

        vnc_flush(vs);

        vnc_read_when(vs, protocol_client_vencrypt_auth, 4);

    }

    return 0;

}
