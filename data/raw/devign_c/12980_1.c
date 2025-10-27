void hmp_savevm(Monitor *mon, const QDict *qdict)
{
    BlockDriverState *bs, *bs1;
    QEMUSnapshotInfo sn1, *sn = &sn1, old_sn1, *old_sn = &old_sn1;
    int ret;
    QEMUFile *f;
    int saved_vm_running;
    uint64_t vm_state_size;
    qemu_timeval tv;
    struct tm tm;
    const char *name = qdict_get_try_str(qdict, "name");
    Error *local_err = NULL;
    /* Verify if there is a device that doesn't support snapshots and is writable */
    bs = NULL;
    while ((bs = bdrv_next(bs))) {
        if (!bdrv_is_inserted(bs) || bdrv_is_read_only(bs)) {
            continue;
        if (!bdrv_can_snapshot(bs)) {
            monitor_printf(mon, "Device '%s' is writable but does not support snapshots.\n",
                               bdrv_get_device_name(bs));
    bs = find_vmstate_bs();
    if (!bs) {
        monitor_printf(mon, "No block device can accept snapshots\n");
    saved_vm_running = runstate_is_running();
    vm_stop(RUN_STATE_SAVE_VM);
    memset(sn, 0, sizeof(*sn));
    /* fill auxiliary fields */
    qemu_gettimeofday(&tv);
    sn->date_sec = tv.tv_sec;
    sn->date_nsec = tv.tv_usec * 1000;
    sn->vm_clock_nsec = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    if (name) {
        ret = bdrv_snapshot_find(bs, old_sn, name);
        if (ret >= 0) {
            pstrcpy(sn->name, sizeof(sn->name), old_sn->name);
            pstrcpy(sn->id_str, sizeof(sn->id_str), old_sn->id_str);
        } else {
            pstrcpy(sn->name, sizeof(sn->name), name);
    } else {
        /* cast below needed for OpenBSD where tv_sec is still 'long' */
        localtime_r((const time_t *)&tv.tv_sec, &tm);
        strftime(sn->name, sizeof(sn->name), "vm-%Y%m%d%H%M%S", &tm);
    /* Delete old snapshots of the same name */
    if (name && del_existing_snapshots(mon, name) < 0) {
        goto the_end;
    /* save the VM state */
    f = qemu_fopen_bdrv(bs, 1);
    if (!f) {
        monitor_printf(mon, "Could not open VM state file\n");
        goto the_end;
    ret = qemu_savevm_state(f, &local_err);
    vm_state_size = qemu_ftell(f);
    qemu_fclose(f);
    if (ret < 0) {
        monitor_printf(mon, "%s\n", error_get_pretty(local_err));
        error_free(local_err);
        goto the_end;
    /* create the snapshots */
    bs1 = NULL;
    while ((bs1 = bdrv_next(bs1))) {
        if (bdrv_can_snapshot(bs1)) {
            /* Write VM state size only to the image that contains the state */
            sn->vm_state_size = (bs == bs1 ? vm_state_size : 0);
            ret = bdrv_snapshot_create(bs1, sn);
            if (ret < 0) {
                monitor_printf(mon, "Error while creating snapshot on '%s'\n",
                               bdrv_get_device_name(bs1));
 the_end:
    if (saved_vm_running) {
        vm_start();