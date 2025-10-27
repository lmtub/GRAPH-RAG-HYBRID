static inline void vhost_dev_log_resize(struct vhost_dev* dev, uint64_t size)

{

    vhost_log_chunk_t *log;

    uint64_t log_base;

    int r;

    if (size) {

        log = g_malloc0(size * sizeof *log);

    } else {

        log = NULL;

    }

    log_base = (uint64_t)(unsigned long)log;

    r = ioctl(dev->control, VHOST_SET_LOG_BASE, &log_base);

    assert(r >= 0);

    vhost_client_sync_dirty_bitmap(&dev->client, 0,

                                   (target_phys_addr_t)~0x0ull);

    if (dev->log) {

        g_free(dev->log);

    }

    dev->log = log;

    dev->log_size = size;

}
