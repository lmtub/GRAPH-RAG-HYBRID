static unsigned syborg_virtio_get_features(void *opaque)

{

    unsigned ret = 0;

    ret |= (1 << VIRTIO_F_NOTIFY_ON_EMPTY);

    return ret;

}
