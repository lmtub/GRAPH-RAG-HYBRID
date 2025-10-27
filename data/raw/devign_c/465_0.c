int vhost_dev_init(struct vhost_dev *hdev, int devfd, const char *devpath,

                   bool force)

{

    uint64_t features;

    int r;

    if (devfd >= 0) {

        hdev->control = devfd;

    } else {

        hdev->control = open(devpath, O_RDWR);

        if (hdev->control < 0) {

            return -errno;

        }

    }

    r = ioctl(hdev->control, VHOST_SET_OWNER, NULL);

    if (r < 0) {

        goto fail;

    }



    r = ioctl(hdev->control, VHOST_GET_FEATURES, &features);

    if (r < 0) {

        goto fail;

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

    hdev->mem = g_malloc0(offsetof(struct vhost_memory, regions));

    hdev->n_mem_sections = 0;

    hdev->mem_sections = NULL;

    hdev->log = NULL;

    hdev->log_size = 0;

    hdev->log_enabled = false;

    hdev->started = false;

    memory_listener_register(&hdev->memory_listener, NULL);

    hdev->force = force;

    return 0;

fail:

    r = -errno;

    close(hdev->control);

    return r;

}
