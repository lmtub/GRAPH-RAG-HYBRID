int load_snapshot(const char *name, Error **errp)

{

    BlockDriverState *bs, *bs_vm_state;

    QEMUSnapshotInfo sn;

    QEMUFile *f;

    int ret;

    AioContext *aio_context;

    MigrationIncomingState *mis = migration_incoming_get_current();



    if (!bdrv_all_can_snapshot(&bs)) {

        error_setg(errp,

                   "Device '%s' is writable but does not support snapshots",

                   bdrv_get_device_name(bs));

        return -ENOTSUP;

    }

    ret = bdrv_all_find_snapshot(name, &bs);

    if (ret < 0) {

        error_setg(errp,

                   "Device '%s' does not have the requested snapshot '%s'",

                   bdrv_get_device_name(bs), name);

        return ret;

    }



    bs_vm_state = bdrv_all_find_vmstate_bs();

    if (!bs_vm_state) {

        error_setg(errp, "No block device supports snapshots");

        return -ENOTSUP;

    }

    aio_context = bdrv_get_aio_context(bs_vm_state);



    /* Don't even try to load empty VM states */

    aio_context_acquire(aio_context);

    ret = bdrv_snapshot_find(bs_vm_state, &sn, name);

    aio_context_release(aio_context);

    if (ret < 0) {

        return ret;

    } else if (sn.vm_state_size == 0) {

        error_setg(errp, "This is a disk-only snapshot. Revert to it "

                   " offline using qemu-img");

        return -EINVAL;

    }



    /* Flush all IO requests so they don't interfere with the new state.  */

    bdrv_drain_all();



    ret = bdrv_all_goto_snapshot(name, &bs);

    if (ret < 0) {

        error_setg(errp, "Error %d while activating snapshot '%s' on '%s'",

                     ret, name, bdrv_get_device_name(bs));

        return ret;

    }



    /* restore the VM state */

    f = qemu_fopen_bdrv(bs_vm_state, 0);

    if (!f) {

        error_setg(errp, "Could not open VM state file");

        return -EINVAL;

    }



    qemu_system_reset(SHUTDOWN_CAUSE_NONE);

    mis->from_src_file = f;



    aio_context_acquire(aio_context);

    ret = qemu_loadvm_state(f);

    qemu_fclose(f);

    aio_context_release(aio_context);



    migration_incoming_state_destroy();

    if (ret < 0) {

        error_setg(errp, "Error %d while loading VM state", ret);

        return ret;

    }



    return 0;

}
