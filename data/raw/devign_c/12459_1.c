void vnc_display_open(const char *id, Error **errp)

{

    VncDisplay *vs = vnc_display_find(id);

    QemuOpts *opts = qemu_opts_find(&qemu_vnc_opts, id);

    QemuOpts *sopts, *wsopts;

    const char *share, *device_id;

    QemuConsole *con;

    bool password = false;

    bool reverse = false;

    const char *vnc;

    const char *has_to;

    char *h;

    bool has_ipv4 = false;

    bool has_ipv6 = false;

    const char *websocket;

    bool tls = false, x509 = false;

#ifdef CONFIG_VNC_TLS

    const char *path;

#endif

    bool sasl = false;

#ifdef CONFIG_VNC_SASL

    int saslErr;

#endif

#if defined(CONFIG_VNC_TLS) || defined(CONFIG_VNC_SASL)

    int acl = 0;

#endif

    int lock_key_sync = 1;



    if (!vs) {

        error_setg(errp, "VNC display not active");

        return;

    }

    vnc_display_close(vs);



    if (!opts) {

        return;

    }

    vnc = qemu_opt_get(opts, "vnc");

    if (!vnc || strcmp(vnc, "none") == 0) {

        return;

    }



    sopts = qemu_opts_create(&socket_optslist, NULL, 0, &error_abort);

    wsopts = qemu_opts_create(&socket_optslist, NULL, 0, &error_abort);



    h = strrchr(vnc, ':');

    if (h) {

        char *host = g_strndup(vnc, h - vnc);

        qemu_opt_set(sopts, "host", host, &error_abort);

        qemu_opt_set(wsopts, "host", host, &error_abort);

        qemu_opt_set(sopts, "port", h+1, &error_abort);

        g_free(host);

    } else {

        error_setg(errp, "no vnc port specified");

        goto fail;

    }



    has_to = qemu_opt_get(opts, "to");

    has_ipv4 = qemu_opt_get_bool(opts, "ipv4", false);

    has_ipv6 = qemu_opt_get_bool(opts, "ipv6", false);

    if (has_to) {

        qemu_opt_set(sopts, "to", has_to, &error_abort);

        qemu_opt_set(wsopts, "to", has_to, &error_abort);

    }

    if (has_ipv4) {

        qemu_opt_set(sopts, "ipv4", "on", &error_abort);

        qemu_opt_set(wsopts, "ipv4", "on", &error_abort);

    }

    if (has_ipv6) {

        qemu_opt_set(sopts, "ipv6", "on", &error_abort);

        qemu_opt_set(wsopts, "ipv6", "on", &error_abort);

    }



    password = qemu_opt_get_bool(opts, "password", false);

    if (password && fips_get_state()) {

        error_setg(errp,

                   "VNC password auth disabled due to FIPS mode, "

                   "consider using the VeNCrypt or SASL authentication "

                   "methods as an alternative");

        goto fail;

    }



    reverse = qemu_opt_get_bool(opts, "reverse", false);

    lock_key_sync = qemu_opt_get_bool(opts, "lock-key-sync", true);

    sasl = qemu_opt_get_bool(opts, "sasl", false);

#ifndef CONFIG_VNC_SASL

    if (sasl) {

        error_setg(errp, "VNC SASL auth requires cyrus-sasl support");

        goto fail;

    }

#endif /* CONFIG_VNC_SASL */

    tls  = qemu_opt_get_bool(opts, "tls", false);

#ifdef CONFIG_VNC_TLS

    path = qemu_opt_get(opts, "x509");

    if (!path) {

        path = qemu_opt_get(opts, "x509verify");

        if (path) {

            vs->tls.x509verify = true;

        }

    }

    if (path) {

        x509 = true;

        if (vnc_tls_set_x509_creds_dir(vs, path) < 0) {

            error_setg(errp, "Failed to find x509 certificates/keys in %s",

                       path);

            goto fail;

        }

    }

#else /* ! CONFIG_VNC_TLS */

    if (tls) {

        error_setg(errp, "VNC TLS auth requires gnutls support");

        goto fail;

    }

#endif /* ! CONFIG_VNC_TLS */

#if defined(CONFIG_VNC_TLS) || defined(CONFIG_VNC_SASL)

    acl = qemu_opt_get_bool(opts, "acl", false);

#endif



    share = qemu_opt_get(opts, "share");

    if (share) {

        if (strcmp(share, "ignore") == 0) {

            vs->share_policy = VNC_SHARE_POLICY_IGNORE;

        } else if (strcmp(share, "allow-exclusive") == 0) {

            vs->share_policy = VNC_SHARE_POLICY_ALLOW_EXCLUSIVE;

        } else if (strcmp(share, "force-shared") == 0) {

            vs->share_policy = VNC_SHARE_POLICY_FORCE_SHARED;

        } else {

            error_setg(errp, "unknown vnc share= option");

            goto fail;

        }

    } else {

        vs->share_policy = VNC_SHARE_POLICY_ALLOW_EXCLUSIVE;

    }

    vs->connections_limit = qemu_opt_get_number(opts, "connections", 32);



    websocket = qemu_opt_get(opts, "websocket");

    if (websocket) {

#ifdef CONFIG_VNC_WS

        vs->ws_enabled = true;

        qemu_opt_set(wsopts, "port", websocket, &error_abort);

#else /* ! CONFIG_VNC_WS */

        error_setg(errp, "Websockets protocol requires gnutls support");

        goto fail;

#endif /* ! CONFIG_VNC_WS */

    }



#ifdef CONFIG_VNC_JPEG

    vs->lossy = qemu_opt_get_bool(opts, "lossy", false);

#endif

    vs->non_adaptive = qemu_opt_get_bool(opts, "non-adaptive", false);

    /* adaptive updates are only used with tight encoding and

     * if lossy updates are enabled so we can disable all the

     * calculations otherwise */

    if (!vs->lossy) {

        vs->non_adaptive = true;

    }



#ifdef CONFIG_VNC_TLS

    if (acl && x509 && vs->tls.x509verify) {

        char *aclname;



        if (strcmp(vs->id, "default") == 0) {

            aclname = g_strdup("vnc.x509dname");

        } else {

            aclname = g_strdup_printf("vnc.%s.x509dname", vs->id);

        }

        vs->tls.acl = qemu_acl_init(aclname);

        if (!vs->tls.acl) {

            fprintf(stderr, "Failed to create x509 dname ACL\n");

            exit(1);

        }

        g_free(aclname);

    }

#endif

#ifdef CONFIG_VNC_SASL

    if (acl && sasl) {

        char *aclname;



        if (strcmp(vs->id, "default") == 0) {

            aclname = g_strdup("vnc.username");

        } else {

            aclname = g_strdup_printf("vnc.%s.username", vs->id);

        }

        vs->sasl.acl = qemu_acl_init(aclname);

        if (!vs->sasl.acl) {

            fprintf(stderr, "Failed to create username ACL\n");

            exit(1);

        }

        g_free(aclname);

    }

#endif



    vnc_display_setup_auth(vs, password, sasl, tls, x509, websocket);



#ifdef CONFIG_VNC_SASL

    if ((saslErr = sasl_server_init(NULL, "qemu")) != SASL_OK) {

        error_setg(errp, "Failed to initialize SASL auth: %s",

                   sasl_errstring(saslErr, NULL, NULL));

        goto fail;

    }

#endif

    vs->lock_key_sync = lock_key_sync;



    device_id = qemu_opt_get(opts, "display");

    if (device_id) {

        DeviceState *dev;

        int head = qemu_opt_get_number(opts, "head", 0);



        dev = qdev_find_recursive(sysbus_get_default(), device_id);

        if (dev == NULL) {

            error_setg(errp, "Device '%s' not found", device_id);

            goto fail;

        }



        con = qemu_console_lookup_by_device(dev, head);

        if (con == NULL) {

            error_setg(errp, "Device %s is not bound to a QemuConsole",

                       device_id);

            goto fail;

        }

    } else {

        con = NULL;

    }



    if (con != vs->dcl.con) {

        unregister_displaychangelistener(&vs->dcl);

        vs->dcl.con = con;

        register_displaychangelistener(&vs->dcl);

    }



    if (reverse) {

        /* connect to viewer */

        int csock;

        vs->lsock = -1;

#ifdef CONFIG_VNC_WS

        vs->lwebsock = -1;

#endif

        if (strncmp(vnc, "unix:", 5) == 0) {

            csock = unix_connect(vnc+5, errp);

        } else {

            csock = inet_connect(vnc, errp);

        }

        if (csock < 0) {

            goto fail;

        }

        vnc_connect(vs, csock, false, false);

    } else {

        /* listen for connects */

        if (strncmp(vnc, "unix:", 5) == 0) {

            vs->lsock = unix_listen(vnc+5, NULL, 0, errp);

            vs->is_unix = true;

        } else {

            vs->lsock = inet_listen_opts(sopts, 5900, errp);

            if (vs->lsock < 0) {

                goto fail;

            }

#ifdef CONFIG_VNC_WS

            if (vs->ws_enabled) {

                vs->lwebsock = inet_listen_opts(wsopts, 0, errp);

                if (vs->lwebsock < 0) {

                    if (vs->lsock != -1) {

                        close(vs->lsock);

                        vs->lsock = -1;

                    }

                    goto fail;

                }

            }

#endif /* CONFIG_VNC_WS */

        }

        vs->enabled = true;

        qemu_set_fd_handler2(vs->lsock, NULL,

                vnc_listen_regular_read, NULL, vs);

#ifdef CONFIG_VNC_WS

        if (vs->ws_enabled) {

            qemu_set_fd_handler2(vs->lwebsock, NULL,

                    vnc_listen_websocket_read, NULL, vs);

        }

#endif /* CONFIG_VNC_WS */

    }

    qemu_opts_del(sopts);

    qemu_opts_del(wsopts);

    return;



fail:

    qemu_opts_del(sopts);

    qemu_opts_del(wsopts);

    vs->enabled = false;

#ifdef CONFIG_VNC_WS

    vs->ws_enabled = false;

#endif /* CONFIG_VNC_WS */

}
