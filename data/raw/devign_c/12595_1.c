static void virtio_balloon_save(QEMUFile *f, void *opaque)

{

    VirtIOBalloon *s = opaque;



    virtio_save(&s->vdev, f);



    qemu_put_be32(f, s->num_pages);

    qemu_put_be32(f, s->actual);

    qemu_put_buffer(f, (uint8_t *)&s->stats_vq_elem, sizeof(VirtQueueElement));

    qemu_put_buffer(f, (uint8_t *)&s->stats_vq_offset, sizeof(size_t));

    qemu_put_buffer(f, (uint8_t *)&s->stats_callback, sizeof(MonitorCompletion));

    qemu_put_buffer(f, (uint8_t *)&s->stats_opaque_callback_data, sizeof(void));

}
