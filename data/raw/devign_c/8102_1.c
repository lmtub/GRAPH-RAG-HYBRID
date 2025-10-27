void do_savevm(const char *name)

{

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo sn1, *sn = &sn1, old_sn1, *old_sn = &old_sn1;

    int must_delete, ret, i;

    BlockDriverInfo bdi1, *bdi = &bdi1;

    QEMUFile *f;

    int saved_vm_running;

#ifdef _WIN32

    struct _timeb tb;

#else

    struct timeval tv;

#endif



    bs = get_bs_snapshots();

    if (!bs) {

        term_printf("No block device can accept snapshots\n");

        return;

    }



    /* ??? Should this occur after vm_stop?  */

    qemu_aio_flush();



    saved_vm_running = vm_running;

    vm_stop(0);



    must_delete = 0;

    if (name) {

        ret = bdrv_snapshot_find(bs, old_sn, name);

        if (ret >= 0) {

            must_delete = 1;

        }

    }

    memset(sn, 0, sizeof(*sn));

    if (must_delete) {

        pstrcpy(sn->name, sizeof(sn->name), old_sn->name);

        pstrcpy(sn->id_str, sizeof(sn->id_str), old_sn->id_str);

    } else {

        if (name)

            pstrcpy(sn->name, sizeof(sn->name), name);

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



    if (bdrv_get_info(bs, bdi) < 0 || bdi->vm_state_offset <= 0) {

        term_printf("Device %s does not support VM state snapshots\n",

                    bdrv_get_device_name(bs));

        goto the_end;

    }



    /* save the VM state */

    f = qemu_fopen_bdrv(bs, bdi->vm_state_offset, 1);

    if (!f) {

        term_printf("Could not open VM state file\n");

        goto the_end;

    }

    ret = qemu_savevm_state(f);

    sn->vm_state_size = qemu_ftell(f);

    qemu_fclose(f);

    if (ret < 0) {

        term_printf("Error %d while writing VM\n", ret);

        goto the_end;

    }



    /* create the snapshots */



    for(i = 0; i < nb_drives; i++) {

        bs1 = drives_table[i].bdrv;

        if (bdrv_has_snapshot(bs1)) {

            if (must_delete) {

                ret = bdrv_snapshot_delete(bs1, old_sn->id_str);

                if (ret < 0) {

                    term_printf("Error while deleting snapshot on '%s'\n",

                                bdrv_get_device_name(bs1));

                }

            }

            ret = bdrv_snapshot_create(bs1, sn);

            if (ret < 0) {

                term_printf("Error while creating snapshot on '%s'\n",

                            bdrv_get_device_name(bs1));

            }

        }

    }



 the_end:

    if (saved_vm_running)

        vm_start();

}
