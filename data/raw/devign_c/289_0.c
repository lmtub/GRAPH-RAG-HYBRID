VirtIODevice *virtio_blk_init(DeviceState *dev, BlockConf *conf,

                              char **serial)

{

    VirtIOBlock *s;

    int cylinders, heads, secs;

    static int virtio_blk_id;

    DriveInfo *dinfo;



    if (!conf->bs) {

        error_report("virtio-blk-pci: drive property not set");

        return NULL;

    }

    if (!bdrv_is_inserted(conf->bs)) {

        error_report("Device needs media, but drive is empty");

        return NULL;

    }



    if (!*serial) {

        /* try to fall back to value set with legacy -drive serial=... */

        dinfo = drive_get_by_blockdev(conf->bs);

        if (*dinfo->serial) {

            *serial = strdup(dinfo->serial);

        }

    }



    s = (VirtIOBlock *)virtio_common_init("virtio-blk", VIRTIO_ID_BLOCK,

                                          sizeof(struct virtio_blk_config),

                                          sizeof(VirtIOBlock));



    s->vdev.get_config = virtio_blk_update_config;

    s->vdev.get_features = virtio_blk_get_features;

    s->vdev.reset = virtio_blk_reset;

    s->bs = conf->bs;

    s->conf = conf;

    s->serial = *serial;

    s->rq = NULL;

    s->sector_mask = (s->conf->logical_block_size / BDRV_SECTOR_SIZE) - 1;

    bdrv_guess_geometry(s->bs, &cylinders, &heads, &secs);



    s->vq = virtio_add_queue(&s->vdev, 128, virtio_blk_handle_output);



    qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    s->qdev = dev;

    register_savevm(dev, "virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);

    bdrv_set_dev_ops(s->bs, &virtio_block_ops, s);

    bdrv_set_buffer_alignment(s->bs, conf->logical_block_size);



    bdrv_iostatus_enable(s->bs);

    add_boot_device_path(conf->bootindex, dev, "/disk@0,0");



    return &s->vdev;

}
