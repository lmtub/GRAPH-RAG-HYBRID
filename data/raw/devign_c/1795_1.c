static int scsi_disk_initfn(SCSIDevice *dev)
{
    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);
    int is_cd;
    DriveInfo *dinfo;
    if (!s->qdev.conf.bs) {
        error_report("scsi-disk: drive property not set");
    s->bs = s->qdev.conf.bs;
    is_cd = bdrv_get_type_hint(s->bs) == BDRV_TYPE_CDROM;
    if (bdrv_get_on_error(s->bs, 1) != BLOCK_ERR_REPORT) {
        error_report("Device doesn't support drive option rerror");
    if (!s->serial) {
        /* try to fall back to value set with legacy -drive serial=... */
        dinfo = drive_get_by_blockdev(s->bs);
        s->serial = qemu_strdup(*dinfo->serial ? dinfo->serial : "0");
    if (!s->version) {
        s->version = qemu_strdup(QEMU_VERSION);
    if (bdrv_is_sg(s->bs)) {
        error_report("scsi-disk: unwanted /dev/sg*");
    if (is_cd) {
        s->qdev.blocksize = 2048;
    } else {
        s->qdev.blocksize = s->qdev.conf.logical_block_size;
    s->cluster_size = s->qdev.blocksize / 512;
    s->qdev.type = TYPE_DISK;
    qemu_add_vm_change_state_handler(scsi_dma_restart_cb, s);
    bdrv_set_removable(s->bs, is_cd);
    return 0;