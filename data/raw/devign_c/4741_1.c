static int virtio_balloon_load(QEMUFile *f, void *opaque, int version_id)

{

    VirtIOBalloon *s = opaque;



    if (version_id != 1)

        return -EINVAL;



    virtio_load(&s->vdev, f);



    s->num_pages = qemu_get_be32(f);

    s->actual = qemu_get_be32(f);

    qemu_get_buffer(f, (uint8_t *)&s->stats_vq_elem, sizeof(VirtQueueElement));

    qemu_get_buffer(f, (uint8_t *)&s->stats_vq_offset, sizeof(size_t));

    qemu_get_buffer(f, (uint8_t *)&s->stats_callback, sizeof(MonitorCompletion));

    qemu_get_buffer(f, (uint8_t *)&s->stats_opaque_callback_data, sizeof(void));



    return 0;

}
