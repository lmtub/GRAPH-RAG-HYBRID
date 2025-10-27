int nbd_receive_reply(QIOChannel *ioc, NBDReply *reply, Error **errp)

{

    int ret;

    const char *type;



    ret = nbd_read_eof(ioc, &reply->magic, sizeof(reply->magic), errp);

    if (ret <= 0) {

        return ret;

    }



    be32_to_cpus(&reply->magic);



    switch (reply->magic) {

    case NBD_SIMPLE_REPLY_MAGIC:

        ret = nbd_receive_simple_reply(ioc, &reply->simple, errp);

        if (ret < 0) {

            break;

        }



        trace_nbd_receive_simple_reply(reply->simple.error,

                                       nbd_err_lookup(reply->simple.error),

                                       reply->handle);

        if (reply->simple.error == NBD_ESHUTDOWN) {

            /* This works even on mingw which lacks a native ESHUTDOWN */

            error_setg(errp, "server shutting down");

            return -EINVAL;

        }

        break;

    case NBD_STRUCTURED_REPLY_MAGIC:

        ret = nbd_receive_structured_reply_chunk(ioc, &reply->structured, errp);

        if (ret < 0) {

            break;

        }

        type = nbd_reply_type_lookup(reply->structured.type);

        trace_nbd_receive_structured_reply_chunk(reply->structured.flags,

                                                 reply->structured.type, type,

                                                 reply->structured.handle,

                                                 reply->structured.length);

        break;

    default:

        error_setg(errp, "invalid magic (got 0x%" PRIx32 ")", reply->magic);

        return -EINVAL;

    }

    if (ret < 0) {

        return ret;

    }



    return 1;

}
