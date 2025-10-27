void virtio_cleanup(VirtIODevice *vdev)

{

    qemu_del_vm_change_state_handler(vdev->vmstate);

    g_free(vdev->config);

    g_free(vdev->vq);

    g_free(vdev->vector_queues);

}
