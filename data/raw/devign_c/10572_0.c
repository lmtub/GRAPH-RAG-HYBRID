int nbd_receive_negotiate(QIOChannel *ioc, const char *name, uint16_t *flags,

                          QCryptoTLSCreds *tlscreds, const char *hostname,

                          QIOChannel **outioc,

                          off_t *size, Error **errp)

{

    char buf[256];

    uint64_t magic, s;

    int rc;

    bool zeroes = true;



    TRACE("Receiving negotiation tlscreds=%p hostname=%s.",

          tlscreds, hostname ? hostname : "<null>");



    rc = -EINVAL;



    if (outioc) {

        *outioc = NULL;

    }

    if (tlscreds && !outioc) {

        error_setg(errp, "Output I/O channel required for TLS");

        goto fail;

    }



    if (read_sync(ioc, buf, 8, errp) < 0) {

        error_prepend(errp, "Failed to read data");

        goto fail;

    }



    buf[8] = '\0';

    if (strlen(buf) == 0) {

        error_setg(errp, "Server connection closed unexpectedly");

        goto fail;

    }



    TRACE("Magic is %c%c%c%c%c%c%c%c",

          qemu_isprint(buf[0]) ? buf[0] : '.',

          qemu_isprint(buf[1]) ? buf[1] : '.',

          qemu_isprint(buf[2]) ? buf[2] : '.',

          qemu_isprint(buf[3]) ? buf[3] : '.',

          qemu_isprint(buf[4]) ? buf[4] : '.',

          qemu_isprint(buf[5]) ? buf[5] : '.',

          qemu_isprint(buf[6]) ? buf[6] : '.',

          qemu_isprint(buf[7]) ? buf[7] : '.');



    if (memcmp(buf, "NBDMAGIC", 8) != 0) {

        error_setg(errp, "Invalid magic received");

        goto fail;

    }



    if (read_sync(ioc, &magic, sizeof(magic), errp) < 0) {

        error_prepend(errp, "Failed to read magic");

        goto fail;

    }

    magic = be64_to_cpu(magic);

    TRACE("Magic is 0x%" PRIx64, magic);



    if (magic == NBD_OPTS_MAGIC) {

        uint32_t clientflags = 0;

        uint16_t globalflags;

        bool fixedNewStyle = false;



        if (read_sync(ioc, &globalflags, sizeof(globalflags), errp) < 0) {

            error_prepend(errp, "Failed to read server flags");

            goto fail;

        }

        globalflags = be16_to_cpu(globalflags);

        TRACE("Global flags are %" PRIx32, globalflags);

        if (globalflags & NBD_FLAG_FIXED_NEWSTYLE) {

            fixedNewStyle = true;

            TRACE("Server supports fixed new style");

            clientflags |= NBD_FLAG_C_FIXED_NEWSTYLE;

        }

        if (globalflags & NBD_FLAG_NO_ZEROES) {

            zeroes = false;

            TRACE("Server supports no zeroes");

            clientflags |= NBD_FLAG_C_NO_ZEROES;

        }

        /* client requested flags */

        clientflags = cpu_to_be32(clientflags);

        if (write_sync(ioc, &clientflags, sizeof(clientflags), errp) < 0) {

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

            TRACE("Using default NBD export name \"\"");

            name = "";

        }

        if (fixedNewStyle) {

            /* Check our desired export is present in the

             * server export list. Since NBD_OPT_EXPORT_NAME

             * cannot return an error message, running this

             * query gives us good error reporting if the

             * server required TLS

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

        if (read_sync(ioc, &s, sizeof(s), errp) < 0) {

            error_prepend(errp, "Failed to read export length");

            goto fail;

        }

        *size = be64_to_cpu(s);



        if (read_sync(ioc, flags, sizeof(*flags), errp) < 0) {

            error_prepend(errp, "Failed to read export flags");

            goto fail;

        }

        be16_to_cpus(flags);

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



        if (read_sync(ioc, &s, sizeof(s), errp) < 0) {

            error_prepend(errp, "Failed to read export length");

            goto fail;

        }

        *size = be64_to_cpu(s);

        TRACE("Size is %" PRIu64, *size);



        if (read_sync(ioc, &oldflags, sizeof(oldflags), errp) < 0) {

            error_prepend(errp, "Failed to read export flags");

            goto fail;

        }

        be32_to_cpus(&oldflags);

        if (oldflags & ~0xffff) {

            error_setg(errp, "Unexpected export flags %0x" PRIx32, oldflags);

            goto fail;

        }

        *flags = oldflags;

    } else {

        error_setg(errp, "Bad magic received");

        goto fail;

    }



    TRACE("Size is %" PRIu64 ", export flags %" PRIx16, *size, *flags);

    if (zeroes && drop_sync(ioc, 124, errp) < 0) {

        error_prepend(errp, "Failed to read reserved block");

        goto fail;

    }

    rc = 0;



fail:

    return rc;

}
