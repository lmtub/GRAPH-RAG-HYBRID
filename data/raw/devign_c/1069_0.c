static int nbd_send_negotiate(NBDClient *client)

{

    int csock = client->sock;

    char buf[8 + 8 + 8 + 128];

    int rc;

    const int myflags = (NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_TRIM |

                         NBD_FLAG_SEND_FLUSH | NBD_FLAG_SEND_FUA);



    /* Negotiation header without options:

        [ 0 ..   7]   passwd       ("NBDMAGIC")

        [ 8 ..  15]   magic        (NBD_CLIENT_MAGIC)

        [16 ..  23]   size

        [24 ..  25]   server flags (0)

        [26 ..  27]   export flags

        [28 .. 151]   reserved     (0)



       Negotiation header with options, part 1:

        [ 0 ..   7]   passwd       ("NBDMAGIC")

        [ 8 ..  15]   magic        (NBD_OPTS_MAGIC)

        [16 ..  17]   server flags (0)



       part 2 (after options are sent):

        [18 ..  25]   size

        [26 ..  27]   export flags

        [28 .. 151]   reserved     (0)

     */



    qemu_set_block(csock);

    rc = -EINVAL;



    TRACE("Beginning negotiation.");

    memset(buf, 0, sizeof(buf));

    memcpy(buf, "NBDMAGIC", 8);

    if (client->exp) {

        assert ((client->exp->nbdflags & ~65535) == 0);

        cpu_to_be64w((uint64_t*)(buf + 8), NBD_CLIENT_MAGIC);

        cpu_to_be64w((uint64_t*)(buf + 16), client->exp->size);

        cpu_to_be16w((uint16_t*)(buf + 26), client->exp->nbdflags | myflags);

    } else {

        cpu_to_be64w((uint64_t*)(buf + 8), NBD_OPTS_MAGIC);

        cpu_to_be16w((uint16_t *)(buf + 16), NBD_FLAG_FIXED_NEWSTYLE);

    }



    if (client->exp) {

        if (write_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

            LOG("write failed");

            goto fail;

        }

    } else {

        if (write_sync(csock, buf, 18) != 18) {

            LOG("write failed");

            goto fail;

        }

        rc = nbd_receive_options(client);

        if (rc != 0) {

            LOG("option negotiation failed");

            goto fail;

        }



        assert ((client->exp->nbdflags & ~65535) == 0);

        cpu_to_be64w((uint64_t*)(buf + 18), client->exp->size);

        cpu_to_be16w((uint16_t*)(buf + 26), client->exp->nbdflags | myflags);

        if (write_sync(csock, buf + 18, sizeof(buf) - 18) != sizeof(buf) - 18) {

            LOG("write failed");

            goto fail;

        }

    }



    TRACE("Negotiation succeeded.");

    rc = 0;

fail:

    qemu_set_nonblock(csock);

    return rc;

}
