int load_vmstate(Monitor *mon, const char *name)

{

    DriveInfo *dinfo;

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo sn;

    QEMUFile *f;

    int ret;



    bs = get_bs_snapshots();

    if (!bs) {

        monitor_printf(mon, "No block device supports snapshots\n");

        return -EINVAL;

    }



    /* Flush all IO requests so they don't interfere with the new state.  */

    qemu_aio_flush();



    TAILQ_FOREACH(dinfo, &drives, next) {

        bs1 = dinfo->bdrv;

        if (bdrv_has_snapshot(bs1)) {

            ret = bdrv_snapshot_goto(bs1, name);

            if (ret < 0) {

                if (bs != bs1)

                    monitor_printf(mon, "Warning: ");

                switch(ret) {

                case -ENOTSUP:

                    monitor_printf(mon,

                                   "Snapshots not supported on device '%s'\n",

                                   bdrv_get_device_name(bs1));

                    break;

                case -ENOENT:

                    monitor_printf(mon, "Could not find snapshot '%s' on "

                                   "device '%s'\n",

                                   name, bdrv_get_device_name(bs1));

                    break;

                default:

                    monitor_printf(mon, "Error %d while activating snapshot on"

                                   " '%s'\n", ret, bdrv_get_device_name(bs1));

                    break;

                }

                /* fatal on snapshot block device */

                if (bs == bs1)

                    return 0;

            }

        }

    }



    /* Don't even try to load empty VM states */

    ret = bdrv_snapshot_find(bs, &sn, name);

    if ((ret >= 0) && (sn.vm_state_size == 0))

        return -EINVAL;



    /* restore the VM state */

    f = qemu_fopen_bdrv(bs, 0);

    if (!f) {

        monitor_printf(mon, "Could not open VM state file\n");

        return -EINVAL;

    }

    ret = qemu_loadvm_state(f);

    qemu_fclose(f);

    if (ret < 0) {

        monitor_printf(mon, "Error %d while loading VM state\n", ret);

        return ret;

    }

    return 0;

}
