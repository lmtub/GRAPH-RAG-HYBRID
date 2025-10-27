VirtIODevice *virtio_blk_init(DeviceState *dev, BlockConf *conf)

{

    VirtIOBlock *s;

    int cylinders, heads, secs;

    static int virtio_blk_id;



    s = (VirtIOBlock *)virtio_common_init("virtio-blk", VIRTIO_ID_BLOCK,

                                          sizeof(struct virtio_blk_config),

                                          sizeof(VirtIOBlock));



    s->vdev.get_config = virtio_blk_update_config;

    s->vdev.get_features = virtio_blk_get_features;

    s->vdev.reset = virtio_blk_reset;

    s->bs = conf->dinfo->bdrv;

    s->conf = conf;

    s->rq = NULL;

    s->sector_mask = (s->conf->logical_block_size / BDRV_SECTOR_SIZE) - 1;

    bdrv_guess_geometry(s->bs, &cylinders, &heads, &secs);



    s->vq = virtio_add_queue(&s->vdev, 128, virtio_blk_handle_output);



    qemu_add_vm_change_state_handler(virtio_blk_dma_restart_cb, s);

    register_savevm("virtio-blk", virtio_blk_id++, 2,

                    virtio_blk_save, virtio_blk_load, s);



    return &s->vdev;

}
