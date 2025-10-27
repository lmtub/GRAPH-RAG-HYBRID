static int nbd_negotiate_options(NBDClient *client, Error **errp)

{

    uint32_t flags;

    bool fixedNewstyle = false;



    /* Client sends:

        [ 0 ..   3]   client flags



        [ 0 ..   7]   NBD_OPTS_MAGIC

        [ 8 ..  11]   NBD option

        [12 ..  15]   Data length

        ...           Rest of request



        [ 0 ..   7]   NBD_OPTS_MAGIC

        [ 8 ..  11]   Second NBD option

        [12 ..  15]   Data length

        ...           Rest of request

    */



    if (nbd_read(client->ioc, &flags, sizeof(flags), errp) < 0) {

        error_prepend(errp, "read failed: ");

        return -EIO;

    }

    trace_nbd_negotiate_options_flags();

    be32_to_cpus(&flags);

    if (flags & NBD_FLAG_C_FIXED_NEWSTYLE) {

        trace_nbd_negotiate_options_newstyle();

        fixedNewstyle = true;

        flags &= ~NBD_FLAG_C_FIXED_NEWSTYLE;

    }

    if (flags & NBD_FLAG_C_NO_ZEROES) {

        trace_nbd_negotiate_options_no_zeroes();

        client->no_zeroes = true;

        flags &= ~NBD_FLAG_C_NO_ZEROES;

    }

    if (flags != 0) {

        error_setg(errp, "Unknown client flags 0x%" PRIx32 " received", flags);

        return -EIO;

    }



    while (1) {

        int ret;

        uint32_t option, length;

        uint64_t magic;



        if (nbd_read(client->ioc, &magic, sizeof(magic), errp) < 0) {

            error_prepend(errp, "read failed: ");

            return -EINVAL;

        }

        magic = be64_to_cpu(magic);

        trace_nbd_negotiate_options_check_magic(magic);

        if (magic != NBD_OPTS_MAGIC) {

            error_setg(errp, "Bad magic received");

            return -EINVAL;

        }



        if (nbd_read(client->ioc, &option,

                     sizeof(option), errp) < 0) {

            error_prepend(errp, "read failed: ");

            return -EINVAL;

        }

        option = be32_to_cpu(option);



        if (nbd_read(client->ioc, &length, sizeof(length), errp) < 0) {

            error_prepend(errp, "read failed: ");

            return -EINVAL;

        }

        length = be32_to_cpu(length);



        trace_nbd_negotiate_options_check_option(option);

        if (client->tlscreds &&

            client->ioc == (QIOChannel *)client->sioc) {

            QIOChannel *tioc;

            if (!fixedNewstyle) {

                error_setg(errp, "Unsupported option 0x%" PRIx32, option);

                return -EINVAL;

            }

            switch (option) {

            case NBD_OPT_STARTTLS:

                tioc = nbd_negotiate_handle_starttls(client, length, errp);

                if (!tioc) {

                    return -EIO;

                }

                object_unref(OBJECT(client->ioc));

                client->ioc = QIO_CHANNEL(tioc);

                break;



            case NBD_OPT_EXPORT_NAME:

                /* No way to return an error to client, so drop connection */

                error_setg(errp, "Option 0x%x not permitted before TLS",

                           option);

                return -EINVAL;



            default:

                if (nbd_drop(client->ioc, length, errp) < 0) {

                    return -EIO;

                }

                ret = nbd_negotiate_send_rep_err(client->ioc,

                                                 NBD_REP_ERR_TLS_REQD,

                                                 option, errp,

                                                 "Option 0x%" PRIx32

                                                 "not permitted before TLS",

                                                 option);

                if (ret < 0) {

                    return ret;

                }

                /* Let the client keep trying, unless they asked to

                 * quit. In this mode, we've already sent an error, so

                 * we can't ack the abort.  */

                if (option == NBD_OPT_ABORT) {

                    return 1;

                }

                break;

            }

        } else if (fixedNewstyle) {

            switch (option) {

            case NBD_OPT_LIST:

                ret = nbd_negotiate_handle_list(client, length, errp);

                if (ret < 0) {

                    return ret;

                }

                break;



            case NBD_OPT_ABORT:

                /* NBD spec says we must try to reply before

                 * disconnecting, but that we must also tolerate

                 * guests that don't wait for our reply. */

                nbd_negotiate_send_rep(client->ioc, NBD_REP_ACK, option, NULL);

                return 1;



            case NBD_OPT_EXPORT_NAME:

                return nbd_negotiate_handle_export_name(client, length, errp);



            case NBD_OPT_STARTTLS:

                if (nbd_drop(client->ioc, length, errp) < 0) {

                    return -EIO;

                }

                if (client->tlscreds) {

                    ret = nbd_negotiate_send_rep_err(client->ioc,

                                                     NBD_REP_ERR_INVALID,

                                                     option, errp,

                                                     "TLS already enabled");

                } else {

                    ret = nbd_negotiate_send_rep_err(client->ioc,

                                                     NBD_REP_ERR_POLICY,

                                                     option, errp,

                                                     "TLS not configured");

                }

                if (ret < 0) {

                    return ret;

                }

                break;

            default:

                if (nbd_drop(client->ioc, length, errp) < 0) {

                    return -EIO;

                }

                ret = nbd_negotiate_send_rep_err(client->ioc,

                                                 NBD_REP_ERR_UNSUP,

                                                 option, errp,

                                                 "Unsupported option 0x%"

                                                 PRIx32,

                                                 option);

                if (ret < 0) {

                    return ret;

                }

                break;

            }

        } else {

            /*

             * If broken new-style we should drop the connection

             * for anything except NBD_OPT_EXPORT_NAME

             */

            switch (option) {

            case NBD_OPT_EXPORT_NAME:

                return nbd_negotiate_handle_export_name(client, length, errp);



            default:

                error_setg(errp, "Unsupported option 0x%" PRIx32, option);

                return -EINVAL;

            }

        }

    }

}
