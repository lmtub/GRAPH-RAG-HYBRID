static bool balloon_stats_supported(const VirtIOBalloon *s)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(s);

    return vdev->guest_features & (1 << VIRTIO_BALLOON_F_STATS_VQ);

}
