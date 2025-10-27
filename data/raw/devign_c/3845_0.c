static int virtio_net_load(QEMUFile *f, void *opaque, int version_id)

{

    VirtIONet *n = opaque;

    VirtIODevice *vdev = VIRTIO_DEVICE(n);



    if (version_id < 2 || version_id > VIRTIO_NET_VM_VERSION)

        return -EINVAL;



    return virtio_load(vdev, f, version_id);

}
