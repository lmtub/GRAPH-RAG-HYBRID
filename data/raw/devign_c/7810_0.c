static void virtio_blk_set_status(VirtIODevice *vdev, uint8_t status)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    uint32_t features;



    if (s->dataplane && !(status & (VIRTIO_CONFIG_S_DRIVER |

                                    VIRTIO_CONFIG_S_DRIVER_OK))) {

        virtio_blk_data_plane_stop(s->dataplane);

    }



    if (!(status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        return;

    }



    features = vdev->guest_features;



    /* A guest that supports VIRTIO_BLK_F_CONFIG_WCE must be able to send

     * cache flushes.  Thus, the "auto writethrough" behavior is never

     * necessary for guests that support the VIRTIO_BLK_F_CONFIG_WCE feature.

     * Leaving it enabled would break the following sequence:

     *

     *     Guest started with "-drive cache=writethrough"

     *     Guest sets status to 0

     *     Guest sets DRIVER bit in status field

     *     Guest reads host features (WCE=0, CONFIG_WCE=1)

     *     Guest writes guest features (WCE=0, CONFIG_WCE=1)

     *     Guest writes 1 to the WCE configuration field (writeback mode)

     *     Guest sets DRIVER_OK bit in status field

     *

     * s->bs would erroneously be placed in writethrough mode.

     */

    if (!(features & (1 << VIRTIO_BLK_F_CONFIG_WCE))) {

        aio_context_acquire(bdrv_get_aio_context(s->bs));

        bdrv_set_enable_write_cache(s->bs,

                                    !!(features & (1 << VIRTIO_BLK_F_WCE)));

        aio_context_release(bdrv_get_aio_context(s->bs));

    }

}
