static void vhost_dev_sync_region(struct vhost_dev *dev,

                                  MemoryRegionSection *section,

                                  uint64_t mfirst, uint64_t mlast,

                                  uint64_t rfirst, uint64_t rlast)

{

    uint64_t start = MAX(mfirst, rfirst);

    uint64_t end = MIN(mlast, rlast);

    vhost_log_chunk_t *from = dev->log + start / VHOST_LOG_CHUNK;

    vhost_log_chunk_t *to = dev->log + end / VHOST_LOG_CHUNK + 1;

    uint64_t addr = (start / VHOST_LOG_CHUNK) * VHOST_LOG_CHUNK;



    if (end < start) {

        return;

    }

    assert(end / VHOST_LOG_CHUNK < dev->log_size);

    assert(start / VHOST_LOG_CHUNK < dev->log_size);



    for (;from < to; ++from) {

        vhost_log_chunk_t log;

        int bit;

        /* We first check with non-atomic: much cheaper,

         * and we expect non-dirty to be the common case. */

        if (!*from) {

            addr += VHOST_LOG_CHUNK;

            continue;

        }

        /* Data must be read atomically. We don't really

         * need the barrier semantics of __sync

         * builtins, but it's easier to use them than

         * roll our own. */

        log = __sync_fetch_and_and(from, 0);

        while ((bit = sizeof(log) > sizeof(int) ?

                ffsll(log) : ffs(log))) {

            ram_addr_t ram_addr;

            bit -= 1;

            ram_addr = section->offset_within_region + bit * VHOST_LOG_PAGE;

            memory_region_set_dirty(section->mr, ram_addr, VHOST_LOG_PAGE);

            log &= ~(0x1ull << bit);

        }

        addr += VHOST_LOG_CHUNK;

    }

}
