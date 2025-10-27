void do_savevm(Monitor *mon, const QDict *qdict)

{

    DriveInfo *dinfo;

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo sn1, *sn = &sn1, old_sn1, *old_sn = &old_sn1;

    int ret;

    QEMUFile *f;

    int saved_vm_running;

    uint32_t vm_state_size;

#ifdef _WIN32

    struct _timeb tb;

#else

    struct timeval tv;

#endif

    const char *name = qdict_get_try_str(qdict, "name");



    bs = get_bs_snapshots();

    if (!bs) {

        monitor_printf(mon, "No block device can accept snapshots\n");

        return;

    }



    /* ??? Should this occur after vm_stop?  */

    qemu_aio_flush();



    saved_vm_running = vm_running;

    vm_stop(0);



    memset(sn, 0, sizeof(*sn));

    if (name) {

        ret = bdrv_snapshot_find(bs, old_sn, name);

        if (ret >= 0) {

            pstrcpy(sn->name, sizeof(sn->name), old_sn->name);

            pstrcpy(sn->id_str, sizeof(sn->id_str), old_sn->id_str);

        } else {

            pstrcpy(sn->name, sizeof(sn->name), name);

        }

    }



    /* fill auxiliary fields */

#ifdef _WIN32

    _ftime(&tb);

    sn->date_sec = tb.time;

    sn->date_nsec = tb.millitm * 1000000;

#else

    gettimeofday(&tv, NULL);

    sn->date_sec = tv.tv_sec;

    sn->date_nsec = tv.tv_usec * 1000;

#endif

    sn->vm_clock_nsec = qemu_get_clock(vm_clock);



    /* Delete old snapshots of the same name */

    if (name && del_existing_snapshots(mon, name) < 0) {

        goto the_end;

    }



    /* save the VM state */

    f = qemu_fopen_bdrv(bs, 1);

    if (!f) {

        monitor_printf(mon, "Could not open VM state file\n");

        goto the_end;

    }

    ret = qemu_savevm_state(mon, f);

    vm_state_size = qemu_ftell(f);

    qemu_fclose(f);

    if (ret < 0) {

        monitor_printf(mon, "Error %d while writing VM\n", ret);

        goto the_end;

    }



    /* create the snapshots */



    QTAILQ_FOREACH(dinfo, &drives, next) {

        bs1 = dinfo->bdrv;

        if (bdrv_has_snapshot(bs1)) {

            /* Write VM state size only to the image that contains the state */

            sn->vm_state_size = (bs == bs1 ? vm_state_size : 0);

            ret = bdrv_snapshot_create(bs1, sn);

            if (ret < 0) {

                monitor_printf(mon, "Error while creating snapshot on '%s'\n",

                               bdrv_get_device_name(bs1));

            }

        }

    }



 the_end:

    if (saved_vm_running)

        vm_start();

}
