static inline bool vhost_needs_vring_endian(VirtIODevice *vdev)

{

    if (virtio_vdev_has_feature(vdev, VIRTIO_F_VERSION_1)) {

        return false;

    }

#ifdef TARGET_IS_BIENDIAN

#ifdef HOST_WORDS_BIGENDIAN

    return !virtio_is_big_endian(vdev);

#else

    return virtio_is_big_endian(vdev);

#endif

#else

    return false;

#endif

}
