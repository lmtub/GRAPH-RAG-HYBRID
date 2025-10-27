static inline void vring_used_idx_set(VirtQueue *vq, uint16_t val)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingUsed, idx);

    virtio_stw_phys_cached(vq->vdev, &caches->used, pa, val);

    address_space_cache_invalidate(&caches->used, pa, sizeof(val));

    vq->used_idx = val;

}
