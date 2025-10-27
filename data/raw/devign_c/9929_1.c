int vhost_dev_init(struct vhost_dev *hdev, void *opaque,

                   VhostBackendType backend_type, bool force)

{

    uint64_t features;

    int i, r;



    if (vhost_set_backend_type(hdev, backend_type) < 0) {

        close((uintptr_t)opaque);

        return -1;

    }



    if (hdev->vhost_ops->vhost_backend_init(hdev, opaque) < 0) {

        close((uintptr_t)opaque);

        return -errno;

    }



    r = hdev->vhost_ops->vhost_call(hdev, VHOST_SET_OWNER, NULL);

    if (r < 0) {

        goto fail;

    }



    r = hdev->vhost_ops->vhost_call(hdev, VHOST_GET_FEATURES, &features);

    if (r < 0) {

        goto fail;

    }



    for (i = 0; i < hdev->nvqs; ++i) {

        r = vhost_virtqueue_init(hdev, hdev->vqs + i, i);

        if (r < 0) {

            goto fail_vq;

        }

    }

    hdev->features = features;



    hdev->memory_listener = (MemoryListener) {

        .begin = vhost_begin,

        .commit = vhost_commit,

        .region_add = vhost_region_add,

        .region_del = vhost_region_del,

        .region_nop = vhost_region_nop,

        .log_start = vhost_log_start,

        .log_stop = vhost_log_stop,

        .log_sync = vhost_log_sync,

        .log_global_start = vhost_log_global_start,

        .log_global_stop = vhost_log_global_stop,

        .eventfd_add = vhost_eventfd_add,

        .eventfd_del = vhost_eventfd_del,

        .priority = 10

    };

    hdev->migration_blocker = NULL;

    if (!(hdev->features & (0x1ULL << VHOST_F_LOG_ALL))) {

        error_setg(&hdev->migration_blocker,

                   "Migration disabled: vhost lacks VHOST_F_LOG_ALL feature.");

        migrate_add_blocker(hdev->migration_blocker);

    }

    hdev->mem = g_malloc0(offsetof(struct vhost_memory, regions));

    hdev->n_mem_sections = 0;

    hdev->mem_sections = NULL;

    hdev->log = NULL;

    hdev->log_size = 0;

    hdev->log_enabled = false;

    hdev->started = false;

    hdev->memory_changed = false;

    memory_listener_register(&hdev->memory_listener, &address_space_memory);

    hdev->force = force;

    return 0;

fail_vq:

    while (--i >= 0) {

        vhost_virtqueue_cleanup(hdev->vqs + i);

    }

fail:

    r = -errno;

    hdev->vhost_ops->vhost_backend_cleanup(hdev);

    return r;

}
