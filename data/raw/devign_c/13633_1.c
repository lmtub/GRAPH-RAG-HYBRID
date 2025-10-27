static void vhost_client_set_memory(CPUPhysMemoryClient *client,

                                    target_phys_addr_t start_addr,

                                    ram_addr_t size,

                                    ram_addr_t phys_offset,

                                    bool log_dirty)

{

    struct vhost_dev *dev = container_of(client, struct vhost_dev, client);

    ram_addr_t flags = phys_offset & ~TARGET_PAGE_MASK;

    int s = offsetof(struct vhost_memory, regions) +

        (dev->mem->nregions + 1) * sizeof dev->mem->regions[0];

    uint64_t log_size;

    int r;



    dev->mem = g_realloc(dev->mem, s);



    if (log_dirty) {

        flags = IO_MEM_UNASSIGNED;

    }



    assert(size);



    /* Optimize no-change case. At least cirrus_vga does this a lot at this time. */

    if (flags == IO_MEM_RAM) {

        if (!vhost_dev_cmp_memory(dev, start_addr, size,

                                  (uintptr_t)qemu_get_ram_ptr(phys_offset))) {

            /* Region exists with same address. Nothing to do. */

            return;

        }

    } else {

        if (!vhost_dev_find_reg(dev, start_addr, size)) {

            /* Removing region that we don't access. Nothing to do. */

            return;

        }

    }



    vhost_dev_unassign_memory(dev, start_addr, size);

    if (flags == IO_MEM_RAM) {

        /* Add given mapping, merging adjacent regions if any */

        vhost_dev_assign_memory(dev, start_addr, size,

                                (uintptr_t)qemu_get_ram_ptr(phys_offset));

    } else {

        /* Remove old mapping for this memory, if any. */

        vhost_dev_unassign_memory(dev, start_addr, size);

    }



    if (!dev->started) {

        return;

    }



    if (dev->started) {

        r = vhost_verify_ring_mappings(dev, start_addr, size);

        assert(r >= 0);

    }



    if (!dev->log_enabled) {

        r = ioctl(dev->control, VHOST_SET_MEM_TABLE, dev->mem);

        assert(r >= 0);

        return;

    }

    log_size = vhost_get_log_size(dev);

    /* We allocate an extra 4K bytes to log,

     * to reduce the * number of reallocations. */

#define VHOST_LOG_BUFFER (0x1000 / sizeof *dev->log)

    /* To log more, must increase log size before table update. */

    if (dev->log_size < log_size) {

        vhost_dev_log_resize(dev, log_size + VHOST_LOG_BUFFER);

    }

    r = ioctl(dev->control, VHOST_SET_MEM_TABLE, dev->mem);

    assert(r >= 0);

    /* To log less, can only decrease log size after table update. */

    if (dev->log_size > log_size + VHOST_LOG_BUFFER) {

        vhost_dev_log_resize(dev, log_size);

    }

}
