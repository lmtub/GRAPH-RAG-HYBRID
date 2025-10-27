static void drive_uninit(DriveInfo *dinfo)

{

    qemu_opts_del(dinfo->opts);

    bdrv_delete(dinfo->bdrv);


    QTAILQ_REMOVE(&drives, dinfo, next);

    qemu_free(dinfo);

}