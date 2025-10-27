static int nbd_handle_reply_err(QIOChannel *ioc, nbd_opt_reply *reply,

                                Error **errp)

{

    char *msg = NULL;

    int result = -1;



    if (!(reply->type & (1 << 31))) {

        return 1;

    }



    if (reply->length) {

        if (reply->length > NBD_MAX_BUFFER_SIZE) {

            error_setg(errp, "server error 0x%" PRIx32

                       " (%s) message is too long",

                       reply->type, nbd_rep_lookup(reply->type));

            goto cleanup;

        }

        msg = g_malloc(reply->length + 1);

        if (nbd_read(ioc, msg, reply->length, errp) < 0) {

            error_prepend(errp, "failed to read option error 0x%" PRIx32

                          " (%s) message",

                          reply->type, nbd_rep_lookup(reply->type));

            goto cleanup;

        }

        msg[reply->length] = '\0';

    }



    switch (reply->type) {

    case NBD_REP_ERR_UNSUP:

        trace_nbd_reply_err_unsup(reply->option, nbd_opt_lookup(reply->option));

        result = 0;

        goto cleanup;



    case NBD_REP_ERR_POLICY:

        error_setg(errp, "Denied by server for option %" PRIx32 " (%s)",

                   reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_INVALID:

        error_setg(errp, "Invalid data length for option %" PRIx32 " (%s)",

                   reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_PLATFORM:

        error_setg(errp, "Server lacks support for option %" PRIx32 " (%s)",

                   reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_TLS_REQD:

        error_setg(errp, "TLS negotiation required before option %" PRIx32

                   " (%s)", reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_UNKNOWN:

        error_setg(errp, "Requested export not available for option %" PRIx32

                   " (%s)", reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_SHUTDOWN:

        error_setg(errp, "Server shutting down before option %" PRIx32 " (%s)",

                   reply->option, nbd_opt_lookup(reply->option));

        break;



    case NBD_REP_ERR_BLOCK_SIZE_REQD:

        error_setg(errp, "Server requires INFO_BLOCK_SIZE for option %" PRIx32

                   " (%s)", reply->option, nbd_opt_lookup(reply->option));

        break;



    default:

        error_setg(errp, "Unknown error code when asking for option %" PRIx32

                   " (%s)", reply->option, nbd_opt_lookup(reply->option));

        break;

    }



    if (msg) {

        error_append_hint(errp, "%s\n", msg);

    }



 cleanup:

    g_free(msg);

    if (result < 0) {

        nbd_send_opt_abort(ioc);

    }

    return result;

}
