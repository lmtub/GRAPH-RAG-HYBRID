static int virtio_rng_load_device(VirtIODevice *vdev, QEMUFile *f,

                                  int version_id)

{

    /* We may have an element ready but couldn't process it due to a quota

     * limit.  Make sure to try again after live migration when the quota may

     * have been reset.

     */

    virtio_rng_process(VIRTIO_RNG(vdev));



    return 0;

}
