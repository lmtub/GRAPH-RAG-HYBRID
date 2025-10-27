static inline uint16_t vring_avail_ring(VirtQueue *vq, int i)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingAvail, ring[i]);

    return virtio_lduw_phys_cached(vq->vdev, &caches->avail, pa);

}
