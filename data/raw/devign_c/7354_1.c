static int vnc_display_get_addresses(QemuOpts *opts,

                                     bool reverse,

                                     SocketAddress ***retsaddr,

                                     size_t *retnsaddr,

                                     SocketAddress ***retwsaddr,

                                     size_t *retnwsaddr,

                                     Error **errp)

{

    SocketAddress *saddr = NULL;

    SocketAddress *wsaddr = NULL;

    QemuOptsIter addriter;

    const char *addr;

    int to = qemu_opt_get_number(opts, "to", 0);

    bool has_ipv4 = qemu_opt_get(opts, "ipv4");

    bool has_ipv6 = qemu_opt_get(opts, "ipv6");

    bool ipv4 = qemu_opt_get_bool(opts, "ipv4", false);

    bool ipv6 = qemu_opt_get_bool(opts, "ipv6", false);

    size_t i;

    int displaynum = -1;

    int ret = -1;



    *retsaddr = NULL;

    *retnsaddr = 0;

    *retwsaddr = NULL;

    *retnwsaddr = 0;



    addr = qemu_opt_get(opts, "vnc");

    if (addr == NULL || g_str_equal(addr, "none")) {

        ret = 0;

        goto cleanup;

    }

    if (qemu_opt_get(opts, "websocket") &&

        !qcrypto_hash_supports(QCRYPTO_HASH_ALG_SHA1)) {

        error_setg(errp,

                   "SHA1 hash support is required for websockets");

        goto cleanup;

    }



    qemu_opt_iter_init(&addriter, opts, "vnc");

    while ((addr = qemu_opt_iter_next(&addriter)) != NULL) {

        int rv;

        rv = vnc_display_get_address(addr, false, reverse, 0, to,

                                     has_ipv4, has_ipv6,

                                     ipv4, ipv6,

                                     &saddr, errp);

        if (rv < 0) {

            goto cleanup;

        }

        /* Historical compat - first listen address can be used

         * to set the default websocket port

         */

        if (displaynum == -1) {

            displaynum = rv;

        }

        *retsaddr = g_renew(SocketAddress *, *retsaddr, *retnsaddr + 1);

        (*retsaddr)[(*retnsaddr)++] = saddr;

    }



    /* If we had multiple primary displays, we don't do defaults

     * for websocket, and require explicit config instead. */

    if (*retnsaddr > 1) {

        displaynum = -1;

    }



    qemu_opt_iter_init(&addriter, opts, "websocket");

    while ((addr = qemu_opt_iter_next(&addriter)) != NULL) {

        if (vnc_display_get_address(addr, true, reverse, displaynum, to,

                                    has_ipv4, has_ipv6,

                                    ipv4, ipv6,

                                    &wsaddr, errp) < 0) {

            goto cleanup;

        }



        /* Historical compat - if only a single listen address was

         * provided, then this is used to set the default listen

         * address for websocket too

         */

        if (*retnsaddr == 1 &&

            (*retsaddr)[0]->type == SOCKET_ADDRESS_TYPE_INET &&

            wsaddr->type == SOCKET_ADDRESS_TYPE_INET &&

            g_str_equal(wsaddr->u.inet.host, "") &&

            !g_str_equal((*retsaddr)[0]->u.inet.host, "")) {

            g_free(wsaddr->u.inet.host);

            wsaddr->u.inet.host = g_strdup((*retsaddr)[0]->u.inet.host);

        }



        *retwsaddr = g_renew(SocketAddress *, *retwsaddr, *retnwsaddr + 1);

        (*retwsaddr)[(*retnwsaddr)++] = wsaddr;

    }



    ret = 0;

 cleanup:

    if (ret < 0) {

        for (i = 0; i < *retnsaddr; i++) {

            qapi_free_SocketAddress((*retsaddr)[i]);

        }

        g_free(*retsaddr);

        for (i = 0; i < *retnwsaddr; i++) {

            qapi_free_SocketAddress((*retwsaddr)[i]);

        }

        g_free(*retwsaddr);

        *retsaddr = *retwsaddr = NULL;

        *retnsaddr = *retnwsaddr = 0;

    }

    return ret;

}
