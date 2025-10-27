static uint32_t virtio_9p_get_features(VirtIODevice *vdev, uint32_t features)

{

    features |= 1 << VIRTIO_9P_MOUNT_TAG;

    return features;

}
