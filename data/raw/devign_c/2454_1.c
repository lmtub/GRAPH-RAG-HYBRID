int load_vmstate(const char *name)

{

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo sn;

    QEMUFile *f;

    int ret;



    /* Verify if there is a device that doesn't support snapshots and is writable */

    bs = NULL;

    while ((bs = bdrv_next(bs))) {



        if (bdrv_is_removable(bs) || bdrv_is_read_only(bs)) {

            continue;

        }



        if (!bdrv_can_snapshot(bs)) {

            error_report("Device '%s' is writable but does not support snapshots.",

                               bdrv_get_device_name(bs));

            return -ENOTSUP;

        }

    }



    bs = bdrv_snapshots();

    if (!bs) {

        error_report("No block device supports snapshots");

        return -EINVAL;

    }



    /* Flush all IO requests so they don't interfere with the new state.  */

    qemu_aio_flush();



    bs1 = NULL;

    while ((bs1 = bdrv_next(bs1))) {

        if (bdrv_can_snapshot(bs1)) {

            ret = bdrv_snapshot_goto(bs1, name);

            if (ret < 0) {

                switch(ret) {

                case -ENOTSUP:

                    error_report("%sSnapshots not supported on device '%s'",

                                 bs != bs1 ? "Warning: " : "",

                                 bdrv_get_device_name(bs1));

                    break;

                case -ENOENT:

                    error_report("%sCould not find snapshot '%s' on device '%s'",

                                 bs != bs1 ? "Warning: " : "",

                                 name, bdrv_get_device_name(bs1));

                    break;

                default:

                    error_report("%sError %d while activating snapshot on '%s'",

                                 bs != bs1 ? "Warning: " : "",

                                 ret, bdrv_get_device_name(bs1));

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

        error_report("Could not open VM state file");

        return -EINVAL;

    }

    ret = qemu_loadvm_state(f);

    qemu_fclose(f);

    if (ret < 0) {

        error_report("Error %d while loading VM state", ret);

        return ret;

    }

    return 0;

}
