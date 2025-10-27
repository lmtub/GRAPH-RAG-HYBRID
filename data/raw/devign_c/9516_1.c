static int protocol_client_auth_sasl_start_len(VncState *vs, uint8_t *data, size_t len)

{

    uint32_t startlen = read_u32(data, 0);

    VNC_DEBUG("Got client start len %d\n", startlen);

    if (startlen > SASL_DATA_MAX_LEN) {

        VNC_DEBUG("Too much SASL data %d\n", startlen);

        vnc_client_error(vs);

        return -1;

    }



    if (startlen == 0)

        return protocol_client_auth_sasl_start(vs, NULL, 0);



    vnc_read_when(vs, protocol_client_auth_sasl_start, startlen);

    return 0;

}
