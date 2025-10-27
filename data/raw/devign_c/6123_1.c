static int virtio_rng_load(QEMUFile *f, void *opaque, int version_id)

{

    if (version_id != 1) {

        return -EINVAL;

    }

    return virtio_load(VIRTIO_DEVICE(opaque), f, version_id);

}
