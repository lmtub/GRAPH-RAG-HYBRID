static void do_commit(Monitor *mon, const QDict *qdict)

{

    int all_devices;

    DriveInfo *dinfo;

    const char *device = qdict_get_str(qdict, "device");



    all_devices = !strcmp(device, "all");

    TAILQ_FOREACH(dinfo, &drives, next) {

        if (!all_devices)

            if (strcmp(bdrv_get_device_name(dinfo->bdrv), device))

                continue;

        bdrv_commit(dinfo->bdrv);

    }

}
