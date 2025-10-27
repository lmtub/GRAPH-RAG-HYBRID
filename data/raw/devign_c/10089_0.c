static int sector_limits_lun2qemu(int64_t sector, IscsiLun *iscsilun)

{

    int limit = MIN(sector_lun2qemu(sector, iscsilun), INT_MAX / 2 + 1);



    return limit < BDRV_REQUEST_MAX_SECTORS ? limit : 0;

}
