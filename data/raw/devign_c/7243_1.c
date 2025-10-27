static int virtio_blk_device_exit(DeviceState *dev)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIOBlock *s = VIRTIO_BLK(dev);

#ifdef CONFIG_VIRTIO_BLK_DATA_PLANE

    remove_migration_state_change_notifier(&s->migration_state_notifier);

    virtio_blk_data_plane_destroy(s->dataplane);

    s->dataplane = NULL;

#endif

    qemu_del_vm_change_state_handler(s->change);

    unregister_savevm(dev, "virtio-blk", s);

    blockdev_mark_auto_del(s->bs);

    virtio_cleanup(vdev);

    return 0;

}
