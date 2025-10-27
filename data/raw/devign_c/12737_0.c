static SocketAddressLegacy *sd_server_config(QDict *options, Error **errp)

{

    QDict *server = NULL;

    QObject *crumpled_server = NULL;

    Visitor *iv = NULL;

    SocketAddress *saddr_flat = NULL;

    SocketAddressLegacy *saddr = NULL;

    Error *local_err = NULL;



    qdict_extract_subqdict(options, &server, "server.");



    crumpled_server = qdict_crumple(server, errp);

    if (!crumpled_server) {

        goto done;

    }



    /*

     * FIXME .numeric, .to, .ipv4 or .ipv6 don't work with -drive

     * server.type=inet.  .to doesn't matter, it's ignored anyway.

     * That's because when @options come from -blockdev or

     * blockdev_add, members are typed according to the QAPI schema,

     * but when they come from -drive, they're all QString.  The

     * visitor expects the former.

     */

    iv = qobject_input_visitor_new(crumpled_server);

    visit_type_SocketAddress(iv, NULL, &saddr_flat, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto done;

    }



    saddr = socket_address_crumple(saddr_flat);



done:

    qapi_free_SocketAddress(saddr_flat);

    visit_free(iv);

    qobject_decref(crumpled_server);

    QDECREF(server);

    return saddr;

}
