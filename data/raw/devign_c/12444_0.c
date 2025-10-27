static int nbd_receive_option_reply(QIOChannel *ioc, uint32_t opt,

                                    nbd_opt_reply *reply, Error **errp)

{

    QEMU_BUILD_BUG_ON(sizeof(*reply) != 20);

    if (nbd_read(ioc, reply, sizeof(*reply), errp) < 0) {

        error_prepend(errp, "failed to read option reply");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    be64_to_cpus(&reply->magic);

    be32_to_cpus(&reply->option);

    be32_to_cpus(&reply->type);

    be32_to_cpus(&reply->length);



    trace_nbd_receive_option_reply(reply->option, reply->type, reply->length);



    if (reply->magic != NBD_REP_MAGIC) {

        error_setg(errp, "Unexpected option reply magic");

        nbd_send_opt_abort(ioc);

        return -1;

    }

    if (reply->option != opt) {

        error_setg(errp, "Unexpected option type %x expected %x",

                   reply->option, opt);

        nbd_send_opt_abort(ioc);

        return -1;

    }

    return 0;

}
