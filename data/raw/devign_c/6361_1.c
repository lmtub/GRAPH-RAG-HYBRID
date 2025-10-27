VirtIODevice *virtio_blk_init(DeviceState *dev, VirtIOBlkConf *blk)

{

    VirtIOBlock *s;

    static int virtio_blk_id;



    if (!blk->conf.bs) {

        error_report("drive property not set");

        return NULL;

    }

    if (!bdrv_is_inserted(blk->conf.bs)) {

        error_report("Device needs media, but drive is empty");

        return NULL;

    }



    blkconf_serial(&blk->conf, &blk->serial);

    if (blkconf_geometry(&blk->conf, NULL, 65535, 255, 255) < 0) {

        return NULL;

    }



    s = (VirtIOBlock *)virtio_common_init("virtio-blk", VIRTIO_ID_BLOCK,

                                          sizeof(struct virtio_blk_config),

                                          sizeof(VirtIOBlock));



    s->vdev.get_config = virtio_blk_update_config;

    s->vdev.set_config = virtio_blk_set_config;

    s->vdev.get_features = virtio_blk_get_features;

    s->vdev.set_status = virtio_blk_set_status;

    s->vdev.reset = virtio_blk_reset;

    s->bs = blk->conf.bs;

    s->conf = &blk->conf;

    s->blk = blk;

    s->rq = NULL;

    s->sector_mask = (s->conf->logical_block_size / BDRV_SECTOR_SIZE) - 1;



    s->vq = virtio_add_queue(&s->vdev, 128, virtio_blk_handle_output);

#ifdef CONFIG_VIRTIO_BLK_DATA_PLANE

    if (!virtio_blk_data_plane_create(&s->vdev, blk, &s->dataplane)) {

        virtio_cleanup(&s->vdev);

        return NULL;

    }

#endif



    qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    s->qdev = dev;

    register_savevm(dev, "virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);

    bdrv_set_dev_ops(s->bs, &virtio_block_ops, s);

    bdrv_set_buffer_alignment(s->bs, s->conf->logical_block_size);



    bdrv_iostatus_enable(s->bs);

    add_boot_device_path(s->conf->bootindex, dev, "/disk@0,0");



    return &s->vdev;

}
