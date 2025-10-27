static int del_existing_snapshots(Monitor *mon, const char *name)

{

    BlockDriverState *bs;

    QEMUSnapshotInfo sn1, *snapshot = &sn1;

    int ret;



    bs = NULL;

    while ((bs = bdrv_next(bs))) {

        if (bdrv_can_snapshot(bs) &&

            bdrv_snapshot_find(bs, snapshot, name) >= 0)

        {

            ret = bdrv_snapshot_delete(bs, name);

            if (ret < 0) {

                monitor_printf(mon,

                               "Error while deleting snapshot on '%s'\n",

                               bdrv_get_device_name(bs));

                return -1;

            }

        }

    }



    return 0;

}
