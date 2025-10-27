void qemu_ram_free(ram_addr_t addr)

{

    RAMBlock *block;



    /* This assumes the iothread lock is taken here too.  */

    qemu_mutex_lock_ramlist();

    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (addr == block->offset) {

            QTAILQ_REMOVE(&ram_list.blocks, block, next);

            ram_list.mru_block = NULL;

            ram_list.version++;

            if (block->flags & RAM_PREALLOC_MASK) {

                ;

            } else if (xen_enabled()) {

                xen_invalidate_map_cache_entry(block->host);

            } else if (mem_path) {

#if defined (__linux__) && !defined(TARGET_S390X)

                if (block->fd) {

                    munmap(block->host, block->length);

                    close(block->fd);

                } else {

                    qemu_anon_ram_free(block->host, block->length);

                }

#else

                abort();

#endif

            } else {

                qemu_anon_ram_free(block->host, block->length);

            }

            g_free(block);

            break;

        }

    }

    qemu_mutex_unlock_ramlist();



}
