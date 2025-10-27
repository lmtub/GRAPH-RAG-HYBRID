void start_auth_sasl(VncState *vs)

{

    const char *mechlist = NULL;

    sasl_security_properties_t secprops;

    int err;

    char *localAddr, *remoteAddr;

    int mechlistlen;



    VNC_DEBUG("Initialize SASL auth %d\n", vs->csock);



    /* Get local & remote client addresses in form  IPADDR;PORT */

    if (!(localAddr = vnc_socket_local_addr("%s;%s", vs->csock)))

        goto authabort;



    if (!(remoteAddr = vnc_socket_remote_addr("%s;%s", vs->csock))) {

        free(localAddr);

        goto authabort;

    }



    err = sasl_server_new("vnc",

                          NULL, /* FQDN - just delegates to gethostname */

                          NULL, /* User realm */

                          localAddr,

                          remoteAddr,

                          NULL, /* Callbacks, not needed */

                          SASL_SUCCESS_DATA,

                          &vs->sasl.conn);

    free(localAddr);

    free(remoteAddr);

    localAddr = remoteAddr = NULL;



    if (err != SASL_OK) {

        VNC_DEBUG("sasl context setup failed %d (%s)",

                  err, sasl_errstring(err, NULL, NULL));

        vs->sasl.conn = NULL;

        goto authabort;

    }



#ifdef CONFIG_VNC_TLS

    /* Inform SASL that we've got an external SSF layer from TLS/x509 */

    if (vs->vd->auth == VNC_AUTH_VENCRYPT &&

        vs->vd->subauth == VNC_AUTH_VENCRYPT_X509SASL) {

        gnutls_cipher_algorithm_t cipher;

        sasl_ssf_t ssf;



        cipher = gnutls_cipher_get(vs->tls.session);

        if (!(ssf = (sasl_ssf_t)gnutls_cipher_get_key_size(cipher))) {

            VNC_DEBUG("%s", "cannot TLS get cipher size\n");

            sasl_dispose(&vs->sasl.conn);

            vs->sasl.conn = NULL;

            goto authabort;

        }

        ssf *= 8; /* tls key size is bytes, sasl wants bits */



        err = sasl_setprop(vs->sasl.conn, SASL_SSF_EXTERNAL, &ssf);

        if (err != SASL_OK) {

            VNC_DEBUG("cannot set SASL external SSF %d (%s)\n",

                      err, sasl_errstring(err, NULL, NULL));

            sasl_dispose(&vs->sasl.conn);

            vs->sasl.conn = NULL;

            goto authabort;

        }

    } else

#endif /* CONFIG_VNC_TLS */

        vs->sasl.wantSSF = 1;



    memset (&secprops, 0, sizeof secprops);

    /* Inform SASL that we've got an external SSF layer from TLS */

    if (strncmp(vs->vd->display, "unix:", 5) == 0

#ifdef CONFIG_VNC_TLS

        /* Disable SSF, if using TLS+x509+SASL only. TLS without x509

           is not sufficiently strong */

        || (vs->vd->auth == VNC_AUTH_VENCRYPT &&

            vs->vd->subauth == VNC_AUTH_VENCRYPT_X509SASL)

#endif /* CONFIG_VNC_TLS */

        ) {

        /* If we've got TLS or UNIX domain sock, we don't care about SSF */

        secprops.min_ssf = 0;

        secprops.max_ssf = 0;

        secprops.maxbufsize = 8192;

        secprops.security_flags = 0;

    } else {

        /* Plain TCP, better get an SSF layer */

        secprops.min_ssf = 56; /* Good enough to require kerberos */

        secprops.max_ssf = 100000; /* Arbitrary big number */

        secprops.maxbufsize = 8192;

        /* Forbid any anonymous or trivially crackable auth */

        secprops.security_flags =

            SASL_SEC_NOANONYMOUS | SASL_SEC_NOPLAINTEXT;

    }



    err = sasl_setprop(vs->sasl.conn, SASL_SEC_PROPS, &secprops);

    if (err != SASL_OK) {

        VNC_DEBUG("cannot set SASL security props %d (%s)\n",

                  err, sasl_errstring(err, NULL, NULL));

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

        goto authabort;

    }



    err = sasl_listmech(vs->sasl.conn,

                        NULL, /* Don't need to set user */

                        "", /* Prefix */

                        ",", /* Separator */

                        "", /* Suffix */

                        &mechlist,

                        NULL,

                        NULL);

    if (err != SASL_OK) {

        VNC_DEBUG("cannot list SASL mechanisms %d (%s)\n",

                  err, sasl_errdetail(vs->sasl.conn));

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

        goto authabort;

    }

    VNC_DEBUG("Available mechanisms for client: '%s'\n", mechlist);



    if (!(vs->sasl.mechlist = strdup(mechlist))) {

        VNC_DEBUG("Out of memory");

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

        goto authabort;

    }

    mechlistlen = strlen(mechlist);

    vnc_write_u32(vs, mechlistlen);

    vnc_write(vs, mechlist, mechlistlen);

    vnc_flush(vs);



    VNC_DEBUG("Wait for client mechname length\n");

    vnc_read_when(vs, protocol_client_auth_sasl_mechname_len, 4);



    return;



 authabort:

    vnc_client_error(vs);

    return;

}
