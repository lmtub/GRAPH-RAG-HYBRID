void vhost_dev_cleanup(struct vhost_dev *hdev)

{

    int i;

    for (i = 0; i < hdev->nvqs; ++i) {

        vhost_virtqueue_cleanup(hdev->vqs + i);

    }

    memory_listener_unregister(&hdev->memory_listener);

    if (hdev->migration_blocker) {

        migrate_del_blocker(hdev->migration_blocker);

        error_free(hdev->migration_blocker);

    }

    g_free(hdev->mem);

    g_free(hdev->mem_sections);

    hdev->vhost_ops->vhost_backend_cleanup(hdev);


}