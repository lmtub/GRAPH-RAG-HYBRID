int do_drive_del(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *id = qdict_get_str(qdict, "id");

    BlockDriverState *bs;

    BlockDriverState **ptr;

    Property *prop;



    bs = bdrv_find(id);

    if (!bs) {

        qerror_report(QERR_DEVICE_NOT_FOUND, id);

        return -1;

    }

    if (bdrv_in_use(bs)) {

        qerror_report(QERR_DEVICE_IN_USE, id);

        return -1;

    }



    /* quiesce block driver; prevent further io */

    qemu_aio_flush();

    bdrv_flush(bs);

    bdrv_close(bs);



    /* clean up guest state from pointing to host resource by

     * finding and removing DeviceState "drive" property */

    if (bs->peer) {

        for (prop = bs->peer->info->props; prop && prop->name; prop++) {

            if (prop->info->type == PROP_TYPE_DRIVE) {

                ptr = qdev_get_prop_ptr(bs->peer, prop);

                if (*ptr == bs) {

                    bdrv_detach(bs, bs->peer);

                    *ptr = NULL;

                    break;

                }

            }

        }

    }



    /* clean up host side */

    drive_uninit(drive_get_by_blockdev(bs));



    return 0;

}
