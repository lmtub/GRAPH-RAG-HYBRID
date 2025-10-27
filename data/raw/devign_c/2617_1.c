static int protocol_client_auth_sasl_step_len(VncState *vs, uint8_t *data, size_t len)

{

    uint32_t steplen = read_u32(data, 0);

    VNC_DEBUG("Got client step len %d\n", steplen);

    if (steplen > SASL_DATA_MAX_LEN) {

        VNC_DEBUG("Too much SASL data %d\n", steplen);

        vnc_client_error(vs);

        return -1;

    }



    if (steplen == 0)

        return protocol_client_auth_sasl_step(vs, NULL, 0);

    else

        vnc_read_when(vs, protocol_client_auth_sasl_step, steplen);

    return 0;

}
