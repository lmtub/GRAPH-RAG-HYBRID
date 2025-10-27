int nbd_receive_negotiate(QIOChannel *ioc, const char *name,

                          QCryptoTLSCreds *tlscreds, const char *hostname,

                          QIOChannel **outioc, NBDExportInfo *info,

                          Error **errp)

{

    char buf[256];

    uint64_t magic;

    int rc;

    bool zeroes = true;




    trace_nbd_receive_negotiate(tlscreds, hostname ? hostname : "<null>");




    rc = -EINVAL;



    if (outioc) {

        *outioc = NULL;

    }

    if (tlscreds && !outioc) {

        error_setg(errp, "Output I/O channel required for TLS");

        goto fail;

    }



    if (nbd_read(ioc, buf, 8, errp) < 0) {

        error_prepend(errp, "Failed to read data");

        goto fail;

    }



    buf[8] = '\0';

    if (strlen(buf) == 0) {

        error_setg(errp, "Server connection closed unexpectedly");

        goto fail;

    }



    magic = ldq_be_p(buf);

    trace_nbd_receive_negotiate_magic(magic);



    if (memcmp(buf, "NBDMAGIC", 8) != 0) {

        error_setg(errp, "Invalid magic received");

        goto fail;

    }



    if (nbd_read(ioc, &magic, sizeof(magic), errp) < 0) {

        error_prepend(errp, "Failed to read magic");

        goto fail;

    }

    magic = be64_to_cpu(magic);

    trace_nbd_receive_negotiate_magic(magic);



    if (magic == NBD_OPTS_MAGIC) {

        uint32_t clientflags = 0;

        uint16_t globalflags;

        bool fixedNewStyle = false;



        if (nbd_read(ioc, &globalflags, sizeof(globalflags), errp) < 0) {

            error_prepend(errp, "Failed to read server flags");

            goto fail;

        }

        globalflags = be16_to_cpu(globalflags);

        trace_nbd_receive_negotiate_server_flags(globalflags);

        if (globalflags & NBD_FLAG_FIXED_NEWSTYLE) {

            fixedNewStyle = true;

            clientflags |= NBD_FLAG_C_FIXED_NEWSTYLE;

        }

        if (globalflags & NBD_FLAG_NO_ZEROES) {

            zeroes = false;

            clientflags |= NBD_FLAG_C_NO_ZEROES;

        }

        /* client requested flags */

        clientflags = cpu_to_be32(clientflags);

        if (nbd_write(ioc, &clientflags, sizeof(clientflags), errp) < 0) {

            error_prepend(errp, "Failed to send clientflags field");

            goto fail;

        }

        if (tlscreds) {

            if (fixedNewStyle) {

                *outioc = nbd_receive_starttls(ioc, tlscreds, hostname, errp);

                if (!*outioc) {

                    goto fail;

                }

                ioc = *outioc;

            } else {

                error_setg(errp, "Server does not support STARTTLS");

                goto fail;

            }

        }

        if (!name) {

            trace_nbd_receive_negotiate_default_name();

            name = "";

        }

        if (fixedNewStyle) {

            int result;



            if (structured_reply) {

                result = nbd_request_simple_option(ioc,

                                                   NBD_OPT_STRUCTURED_REPLY,

                                                   errp);

                if (result < 0) {

                    goto fail;

                }

                info->structured_reply = result == 1;

            }



            /* Try NBD_OPT_GO first - if it works, we are done (it

             * also gives us a good message if the server requires

             * TLS).  If it is not available, fall back to

             * NBD_OPT_LIST for nicer error messages about a missing

             * export, then use NBD_OPT_EXPORT_NAME.  */

            result = nbd_opt_go(ioc, name, info, errp);

            if (result < 0) {

                goto fail;

            }

            if (result > 0) {

                return 0;

            }

            /* Check our desired export is present in the

             * server export list. Since NBD_OPT_EXPORT_NAME

             * cannot return an error message, running this

             * query gives us better error reporting if the

             * export name is not available.

             */

            if (nbd_receive_query_exports(ioc, name, errp) < 0) {

                goto fail;

            }

        }

        /* write the export name request */

        if (nbd_send_option_request(ioc, NBD_OPT_EXPORT_NAME, -1, name,

                                    errp) < 0) {

            goto fail;

        }



        /* Read the response */

        if (nbd_read(ioc, &info->size, sizeof(info->size), errp) < 0) {

            error_prepend(errp, "Failed to read export length");

            goto fail;

        }

        be64_to_cpus(&info->size);



        if (nbd_read(ioc, &info->flags, sizeof(info->flags), errp) < 0) {

            error_prepend(errp, "Failed to read export flags");

            goto fail;

        }

        be16_to_cpus(&info->flags);

    } else if (magic == NBD_CLIENT_MAGIC) {

        uint32_t oldflags;



        if (name) {

            error_setg(errp, "Server does not support export names");

            goto fail;

        }

        if (tlscreds) {

            error_setg(errp, "Server does not support STARTTLS");

            goto fail;

        }



        if (nbd_read(ioc, &info->size, sizeof(info->size), errp) < 0) {

            error_prepend(errp, "Failed to read export length");

            goto fail;

        }

        be64_to_cpus(&info->size);



        if (nbd_read(ioc, &oldflags, sizeof(oldflags), errp) < 0) {

            error_prepend(errp, "Failed to read export flags");

            goto fail;

        }

        be32_to_cpus(&oldflags);

        if (oldflags & ~0xffff) {

            error_setg(errp, "Unexpected export flags %0x" PRIx32, oldflags);

            goto fail;

        }

        info->flags = oldflags;

    } else {

        error_setg(errp, "Bad magic received");

        goto fail;

    }



    trace_nbd_receive_negotiate_size_flags(info->size, info->flags);

    if (zeroes && nbd_drop(ioc, 124, errp) < 0) {

        error_prepend(errp, "Failed to read reserved block");

        goto fail;

    }

    rc = 0;



fail:

    return rc;

}