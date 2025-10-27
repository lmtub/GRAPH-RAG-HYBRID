bool virtio_blk_data_plane_create(VirtIODevice *vdev, VirtIOBlkConf *blk,
                                  VirtIOBlockDataPlane **dataplane)
{
    VirtIOBlockDataPlane *s;
    int fd;
    *dataplane = NULL;
    if (!blk->data_plane) {
        return true;
    if (blk->scsi) {
        error_report("device is incompatible with x-data-plane, use scsi=off");
    if (blk->config_wce) {
        error_report("device is incompatible with x-data-plane, "
                     "use config-wce=off");
    fd = raw_get_aio_fd(blk->conf.bs);
    if (fd < 0) {
        error_report("drive is incompatible with x-data-plane, "
                     "use format=raw,cache=none,aio=native");
    s = g_new0(VirtIOBlockDataPlane, 1);
    s->vdev = vdev;
    s->fd = fd;
    s->blk = blk;
    /* Prevent block operations that conflict with data plane thread */
    bdrv_set_in_use(blk->conf.bs, 1);
    *dataplane = s;
    return true;