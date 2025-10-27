static int protocol_client_auth_sasl_step(VncState *vs, uint8_t *data, size_t len)

{

    uint32_t datalen = len;

    const char *serverout;

    unsigned int serveroutlen;

    int err;

    char *clientdata = NULL;



    /* NB, distinction of NULL vs "" is *critical* in SASL */

    if (datalen) {

        clientdata = (char*)data;

        clientdata[datalen-1] = '\0'; /* Wire includes '\0', but make sure */

        datalen--; /* Don't count NULL byte when passing to _start() */

    }



    VNC_DEBUG("Step using SASL Data %p (%d bytes)\n",

              clientdata, datalen);

    err = sasl_server_step(vs->sasl.conn,

                           clientdata,

                           datalen,

                           &serverout,

                           &serveroutlen);

    if (err != SASL_OK &&

        err != SASL_CONTINUE) {

        VNC_DEBUG("sasl step failed %d (%s)\n",

                  err, sasl_errdetail(vs->sasl.conn));

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

        goto authabort;

    }



    if (serveroutlen > SASL_DATA_MAX_LEN) {

        VNC_DEBUG("sasl step reply data too long %d\n",

                  serveroutlen);

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

        goto authabort;

    }



    VNC_DEBUG("SASL return data %d bytes, nil; %d\n",

              serveroutlen, serverout ? 0 : 1);



    if (serveroutlen) {

        vnc_write_u32(vs, serveroutlen + 1);

        vnc_write(vs, serverout, serveroutlen + 1);

    } else {

        vnc_write_u32(vs, 0);

    }



    /* Whether auth is complete */

    vnc_write_u8(vs, err == SASL_CONTINUE ? 0 : 1);



    if (err == SASL_CONTINUE) {

        VNC_DEBUG("%s", "Authentication must continue\n");

        /* Wait for step length */

        vnc_read_when(vs, protocol_client_auth_sasl_step_len, 4);

    } else {

        if (!vnc_auth_sasl_check_ssf(vs)) {

            VNC_DEBUG("Authentication rejected for weak SSF %p\n", vs->ioc);

            goto authreject;

        }



        /* Check username whitelist ACL */

        if (vnc_auth_sasl_check_access(vs) < 0) {

            VNC_DEBUG("Authentication rejected for ACL %p\n", vs->ioc);

            goto authreject;

        }



        VNC_DEBUG("Authentication successful %p\n", vs->ioc);

        vnc_write_u32(vs, 0); /* Accept auth */

        /*

         * Delay writing in SSF encoded mode until pending output

         * buffer is written

         */

        if (vs->sasl.runSSF)

            vs->sasl.waitWriteSSF = vs->output.offset;

        start_client_init(vs);

    }



    return 0;



 authreject:

    vnc_write_u32(vs, 1); /* Reject auth */

    vnc_write_u32(vs, sizeof("Authentication failed"));

    vnc_write(vs, "Authentication failed", sizeof("Authentication failed"));

    vnc_flush(vs);

    vnc_client_error(vs);

    return -1;



 authabort:

    vnc_client_error(vs);

    return -1;

}
