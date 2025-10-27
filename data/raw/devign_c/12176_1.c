static uint16_t vring_used_idx(VirtQueue *vq)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingUsed, idx);

    return virtio_lduw_phys_cached(vq->vdev, &caches->used, pa);

}
