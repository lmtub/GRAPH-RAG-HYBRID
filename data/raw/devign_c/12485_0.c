static int nbd_negotiate_handle_export_name(NBDClient *client, uint32_t length)

{

    int rc = -EINVAL;

    char name[256];



    /* Client sends:

        [20 ..  xx]   export name (length bytes)

     */

    TRACE("Checking length");

    if (length > 255) {

        LOG("Bad length received");

        goto fail;

    }

    if (nbd_negotiate_read(client->ioc, name, length) != length) {

        LOG("read failed");

        goto fail;

    }

    name[length] = '\0';



    TRACE("Client requested export '%s'", name);



    client->exp = nbd_export_find(name);

    if (!client->exp) {

        LOG("export not found");

        goto fail;

    }



    QTAILQ_INSERT_TAIL(&client->exp->clients, client, next);

    nbd_export_get(client->exp);

    rc = 0;

fail:

    return rc;

}
