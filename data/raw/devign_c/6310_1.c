void do_info_snapshots(Monitor *mon)

{

    DriveInfo *dinfo;

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo *sn_tab, *sn;

    int nb_sns, i;

    char buf[256];



    bs = get_bs_snapshots();

    if (!bs) {

        monitor_printf(mon, "No available block device supports snapshots\n");

        return;

    }

    monitor_printf(mon, "Snapshot devices:");

    QTAILQ_FOREACH(dinfo, &drives, next) {

        bs1 = dinfo->bdrv;

        if (bdrv_has_snapshot(bs1)) {

            if (bs == bs1)

                monitor_printf(mon, " %s", bdrv_get_device_name(bs1));

        }

    }

    monitor_printf(mon, "\n");



    nb_sns = bdrv_snapshot_list(bs, &sn_tab);

    if (nb_sns < 0) {

        monitor_printf(mon, "bdrv_snapshot_list: error %d\n", nb_sns);

        return;

    }

    monitor_printf(mon, "Snapshot list (from %s):\n",

                   bdrv_get_device_name(bs));

    monitor_printf(mon, "%s\n", bdrv_snapshot_dump(buf, sizeof(buf), NULL));

    for(i = 0; i < nb_sns; i++) {

        sn = &sn_tab[i];

        monitor_printf(mon, "%s\n", bdrv_snapshot_dump(buf, sizeof(buf), sn));

    }

    qemu_free(sn_tab);

}
