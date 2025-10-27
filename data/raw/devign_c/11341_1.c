void vnc_display_open(DisplayState *ds, const char *display, Error **errp)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;

    const char *options;

    int password = 0;

    int reverse = 0;

#ifdef CONFIG_VNC_TLS

    int tls = 0, x509 = 0;

#endif

#ifdef CONFIG_VNC_SASL

    int sasl = 0;

    int saslErr;

#endif

#if defined(CONFIG_VNC_TLS) || defined(CONFIG_VNC_SASL)

    int acl = 0;

#endif

    int lock_key_sync = 1;



    if (!vnc_display) {

        error_setg(errp, "VNC display not active");

        return;

    }

    vnc_display_close(ds);

    if (strcmp(display, "none") == 0)

        return;



    vs->display = g_strdup(display);

    vs->share_policy = VNC_SHARE_POLICY_ALLOW_EXCLUSIVE;



    options = display;

    while ((options = strchr(options, ','))) {

        options++;

        if (strncmp(options, "password", 8) == 0) {

            if (fips_get_state()) {

                error_setg(errp,

                           "VNC password auth disabled due to FIPS mode, "

                           "consider using the VeNCrypt or SASL authentication "

                           "methods as an alternative");

                goto fail;

            }

            password = 1; /* Require password auth */

        } else if (strncmp(options, "reverse", 7) == 0) {

            reverse = 1;

        } else if (strncmp(options, "no-lock-key-sync", 16) == 0) {

            lock_key_sync = 0;

#ifdef CONFIG_VNC_SASL

        } else if (strncmp(options, "sasl", 4) == 0) {

            sasl = 1; /* Require SASL auth */

#endif

#ifdef CONFIG_VNC_WS

        } else if (strncmp(options, "websocket", 9) == 0) {

            char *start, *end;

            vs->websocket = 1;



            /* Check for 'websocket=<port>' */

            start = strchr(options, '=');

            end = strchr(options, ',');

            if (start && (!end || (start < end))) {

                int len = end ? end-(start+1) : strlen(start+1);

                if (len < 6) {

                    /* extract the host specification from display */

                    char  *host = NULL, *port = NULL, *host_end = NULL;

                    port = g_strndup(start + 1, len);



                    /* ipv6 hosts have colons */

                    end = strchr(display, ',');

                    host_end = g_strrstr_len(display, end - display, ":");



                    if (host_end) {

                        host = g_strndup(display, host_end - display + 1);

                    } else {

                        host = g_strndup(":", 1);

                    }

                    vs->ws_display = g_strconcat(host, port, NULL);

                    g_free(host);

                    g_free(port);

                }

            }

#endif /* CONFIG_VNC_WS */

#ifdef CONFIG_VNC_TLS

        } else if (strncmp(options, "tls", 3) == 0) {

            tls = 1; /* Require TLS */

        } else if (strncmp(options, "x509", 4) == 0) {

            char *start, *end;

            x509 = 1; /* Require x509 certificates */

            if (strncmp(options, "x509verify", 10) == 0)

                vs->tls.x509verify = 1; /* ...and verify client certs */



            /* Now check for 'x509=/some/path' postfix

             * and use that to setup x509 certificate/key paths */

            start = strchr(options, '=');

            end = strchr(options, ',');

            if (start && (!end || (start < end))) {

                int len = end ? end-(start+1) : strlen(start+1);

                char *path = g_strndup(start + 1, len);



                VNC_DEBUG("Trying certificate path '%s'\n", path);

                if (vnc_tls_set_x509_creds_dir(vs, path) < 0) {

                    error_setg(errp, "Failed to find x509 certificates/keys in %s", path);

                    g_free(path);

                    goto fail;

                }

                g_free(path);

            } else {

                error_setg(errp, "No certificate path provided");

                goto fail;

            }

#endif

#if defined(CONFIG_VNC_TLS) || defined(CONFIG_VNC_SASL)

        } else if (strncmp(options, "acl", 3) == 0) {

            acl = 1;

#endif

        } else if (strncmp(options, "lossy", 5) == 0) {

            vs->lossy = true;

        } else if (strncmp(options, "non-adaptive", 12) == 0) {

            vs->non_adaptive = true;

        } else if (strncmp(options, "share=", 6) == 0) {

            if (strncmp(options+6, "ignore", 6) == 0) {

                vs->share_policy = VNC_SHARE_POLICY_IGNORE;

            } else if (strncmp(options+6, "allow-exclusive", 15) == 0) {

                vs->share_policy = VNC_SHARE_POLICY_ALLOW_EXCLUSIVE;

            } else if (strncmp(options+6, "force-shared", 12) == 0) {

                vs->share_policy = VNC_SHARE_POLICY_FORCE_SHARED;

            } else {

                error_setg(errp, "unknown vnc share= option");

                goto fail;

            }

        }

    }



#ifdef CONFIG_VNC_TLS

    if (acl && x509 && vs->tls.x509verify) {

        if (!(vs->tls.acl = qemu_acl_init("vnc.x509dname"))) {

            fprintf(stderr, "Failed to create x509 dname ACL\n");

            exit(1);

        }

    }

#endif

#ifdef CONFIG_VNC_SASL

    if (acl && sasl) {

        if (!(vs->sasl.acl = qemu_acl_init("vnc.username"))) {

            fprintf(stderr, "Failed to create username ACL\n");

            exit(1);

        }

    }

#endif



    /*

     * Combinations we support here:

     *

     *  - no-auth                (clear text, no auth)

     *  - password               (clear text, weak auth)

     *  - sasl                   (encrypt, good auth *IF* using Kerberos via GSSAPI)

     *  - tls                    (encrypt, weak anonymous creds, no auth)

     *  - tls + password         (encrypt, weak anonymous creds, weak auth)

     *  - tls + sasl             (encrypt, weak anonymous creds, good auth)

     *  - tls + x509             (encrypt, good x509 creds, no auth)

     *  - tls + x509 + password  (encrypt, good x509 creds, weak auth)

     *  - tls + x509 + sasl      (encrypt, good x509 creds, good auth)

     *

     * NB1. TLS is a stackable auth scheme.

     * NB2. the x509 schemes have option to validate a client cert dname

     */

    if (password) {

#ifdef CONFIG_VNC_TLS

        if (tls) {

            vs->auth = VNC_AUTH_VENCRYPT;

            if (x509) {

                VNC_DEBUG("Initializing VNC server with x509 password auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_X509VNC;

            } else {

                VNC_DEBUG("Initializing VNC server with TLS password auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_TLSVNC;

            }

        } else {

#endif /* CONFIG_VNC_TLS */

            VNC_DEBUG("Initializing VNC server with password auth\n");

            vs->auth = VNC_AUTH_VNC;

#ifdef CONFIG_VNC_TLS

            vs->subauth = VNC_AUTH_INVALID;

        }

#endif /* CONFIG_VNC_TLS */

#ifdef CONFIG_VNC_SASL

    } else if (sasl) {

#ifdef CONFIG_VNC_TLS

        if (tls) {

            vs->auth = VNC_AUTH_VENCRYPT;

            if (x509) {

                VNC_DEBUG("Initializing VNC server with x509 SASL auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_X509SASL;

            } else {

                VNC_DEBUG("Initializing VNC server with TLS SASL auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_TLSSASL;

            }

        } else {

#endif /* CONFIG_VNC_TLS */

            VNC_DEBUG("Initializing VNC server with SASL auth\n");

            vs->auth = VNC_AUTH_SASL;

#ifdef CONFIG_VNC_TLS

            vs->subauth = VNC_AUTH_INVALID;

        }

#endif /* CONFIG_VNC_TLS */

#endif /* CONFIG_VNC_SASL */

    } else {

#ifdef CONFIG_VNC_TLS

        if (tls) {

            vs->auth = VNC_AUTH_VENCRYPT;

            if (x509) {

                VNC_DEBUG("Initializing VNC server with x509 no auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_X509NONE;

            } else {

                VNC_DEBUG("Initializing VNC server with TLS no auth\n");

                vs->subauth = VNC_AUTH_VENCRYPT_TLSNONE;

            }

        } else {

#endif

            VNC_DEBUG("Initializing VNC server with no auth\n");

            vs->auth = VNC_AUTH_NONE;

#ifdef CONFIG_VNC_TLS

            vs->subauth = VNC_AUTH_INVALID;

        }

#endif

    }



#ifdef CONFIG_VNC_SASL

    if ((saslErr = sasl_server_init(NULL, "qemu")) != SASL_OK) {

        error_setg(errp, "Failed to initialize SASL auth: %s",

                   sasl_errstring(saslErr, NULL, NULL));

        goto fail;

    }

#endif

    vs->lock_key_sync = lock_key_sync;



    if (reverse) {

        /* connect to viewer */

        int csock;

        vs->lsock = -1;

#ifdef CONFIG_VNC_WS

        vs->lwebsock = -1;

#endif

        if (strncmp(display, "unix:", 5) == 0) {

            csock = unix_connect(display+5, errp);

        } else {

            csock = inet_connect(display, errp);

        }

        if (csock < 0) {

            goto fail;

        }

        vnc_connect(vs, csock, 0, 0);

    } else {

        /* listen for connects */

        char *dpy;

        dpy = g_malloc(256);

        if (strncmp(display, "unix:", 5) == 0) {

            pstrcpy(dpy, 256, "unix:");

            vs->lsock = unix_listen(display+5, dpy+5, 256-5, errp);

        } else {

            vs->lsock = inet_listen(display, dpy, 256,

                                    SOCK_STREAM, 5900, errp);

            if (vs->lsock < 0) {

                g_free(dpy);

                goto fail;

            }

#ifdef CONFIG_VNC_WS

            if (vs->websocket) {

                if (vs->ws_display) {

                    vs->lwebsock = inet_listen(vs->ws_display, NULL, 256,

                        SOCK_STREAM, 0, errp);

                } else {

                    vs->lwebsock = inet_listen(vs->display, NULL, 256,

                        SOCK_STREAM, 5700, errp);

                }



                if (vs->lwebsock < 0) {

                    if (vs->lsock) {

                        close(vs->lsock);

                        vs->lsock = -1;

                    }

                    g_free(dpy);

                    goto fail;

                }

            }

#endif /* CONFIG_VNC_WS */

        }

        g_free(vs->display);

        vs->display = dpy;

        qemu_set_fd_handler2(vs->lsock, NULL,

                vnc_listen_regular_read, NULL, vs);

#ifdef CONFIG_VNC_WS

        if (vs->websocket) {

            qemu_set_fd_handler2(vs->lwebsock, NULL,

                    vnc_listen_websocket_read, NULL, vs);

        }

#endif /* CONFIG_VNC_WS */

    }

    return;



fail:

    g_free(vs->display);

    vs->display = NULL;

#ifdef CONFIG_VNC_WS

    g_free(vs->ws_display);

    vs->ws_display = NULL;

#endif /* CONFIG_VNC_WS */

}
