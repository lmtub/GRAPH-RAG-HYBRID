void hmp_nbd_server_start(Monitor *mon, const QDict *qdict)

{

    const char *uri = qdict_get_str(qdict, "uri");

    bool writable = qdict_get_try_bool(qdict, "writable", false);

    bool all = qdict_get_try_bool(qdict, "all", false);

    Error *local_err = NULL;

    BlockInfoList *block_list, *info;

    SocketAddressLegacy *addr;



    if (writable && !all) {

        error_setg(&local_err, "-w only valid together with -a");

        goto exit;

    }



    /* First check if the address is valid and start the server.  */

    addr = socket_parse(uri, &local_err);

    if (local_err != NULL) {

        goto exit;

    }



    qmp_nbd_server_start(addr, false, NULL, &local_err);

    qapi_free_SocketAddressLegacy(addr);

    if (local_err != NULL) {

        goto exit;

    }



    if (!all) {

        return;

    }



    /* Then try adding all block devices.  If one fails, close all and

     * exit.

     */

    block_list = qmp_query_block(NULL);



    for (info = block_list; info; info = info->next) {

        if (!info->value->has_inserted) {

            continue;

        }



        qmp_nbd_server_add(info->value->device, true, writable, &local_err);



        if (local_err != NULL) {

            qmp_nbd_server_stop(NULL);

            break;

        }

    }



    qapi_free_BlockInfoList(block_list);



exit:

    hmp_handle_error(mon, &local_err);

}
