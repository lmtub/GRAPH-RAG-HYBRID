static int nbd_receive_list(QIOChannel *ioc, const char *want, bool *match,

                            Error **errp)

{

    nbd_opt_reply reply;

    uint32_t len;

    uint32_t namelen;

    char name[NBD_MAX_NAME_SIZE + 1];

    int error;



    if (nbd_receive_option_reply(ioc, NBD_OPT_LIST, &reply, errp) < 0) {

        return -1;

    }

    error = nbd_handle_reply_err(ioc, &reply, errp);

    if (error <= 0) {

        /* The server did not support NBD_OPT_LIST, so set *match on

         * the assumption that any name will be accepted.  */

        *match = true;

        return error;

    }

    len = reply.length;



    if (reply.type == NBD_REP_ACK) {

        if (len != 0) {

            error_setg(errp, "length too long for option end");

            nbd_send_opt_abort(ioc);

            return -1;

        }

        return 0;

    } else if (reply.type != NBD_REP_SERVER) {

        error_setg(errp, "Unexpected reply type %" PRIx32 " expected %x",

                   reply.type, NBD_REP_SERVER);

        nbd_send_opt_abort(ioc);

        return -1;

    }



    if (len < sizeof(namelen) || len > NBD_MAX_BUFFER_SIZE) {

        error_setg(errp, "incorrect option length %" PRIu32, len);

        nbd_send_opt_abort(ioc);

        return -1;

    }

    if (read_sync(ioc, &namelen, sizeof(namelen), errp) < 0) {

        error_prepend(errp, "failed to read option name length");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    namelen = be32_to_cpu(namelen);

    len -= sizeof(namelen);

    if (len < namelen) {

        error_setg(errp, "incorrect option name length");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    if (namelen != strlen(want)) {

        if (drop_sync(ioc, len, errp) < 0) {

            error_prepend(errp, "failed to skip export name with wrong length");

            nbd_send_opt_abort(ioc);

            return -1;

        }

        return 1;

    }



    assert(namelen < sizeof(name));

    if (read_sync(ioc, name, namelen, errp) < 0) {

        error_prepend(errp, "failed to read export name");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    name[namelen] = '\0';

    len -= namelen;

    if (drop_sync(ioc, len, errp) < 0) {

        error_prepend(errp, "failed to read export description");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    if (!strcmp(name, want)) {

        *match = true;

    }

    return 1;

}
