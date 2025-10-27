static bool use_multiport(VirtIOSerial *vser)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(vser);

    return virtio_has_feature(vdev, VIRTIO_CONSOLE_F_MULTIPORT);

}
