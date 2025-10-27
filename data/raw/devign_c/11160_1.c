static void virtio_device_free_virtqueues(VirtIODevice *vdev)

{

    int i;

    if (!vdev->vq) {

        return;

    }



    for (i = 0; i < VIRTIO_QUEUE_MAX; i++) {

        VRingMemoryRegionCaches *caches;

        if (vdev->vq[i].vring.num == 0) {

            break;

        }

        caches = atomic_read(&vdev->vq[i].vring.caches);

        atomic_set(&vdev->vq[i].vring.caches, NULL);

        virtio_free_region_cache(caches);

    }

    g_free(vdev->vq);

}
