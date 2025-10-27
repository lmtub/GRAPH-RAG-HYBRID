ram_addr_t qemu_ram_alloc_from_ptr(ram_addr_t size, void *host,

                                   MemoryRegion *mr)

{

    RAMBlock *new_block;



    size = TARGET_PAGE_ALIGN(size);

    new_block = g_malloc0(sizeof(*new_block));



    new_block->mr = mr;

    new_block->offset = find_ram_offset(size);

    if (host) {

        new_block->host = host;

        new_block->flags |= RAM_PREALLOC_MASK;

    } else {

        if (mem_path) {

#if defined (__linux__) && !defined(TARGET_S390X)

            new_block->host = file_ram_alloc(new_block, size, mem_path);

            if (!new_block->host) {

                new_block->host = qemu_vmalloc(size);

                qemu_madvise(new_block->host, size, QEMU_MADV_MERGEABLE);

            }

#else

            fprintf(stderr, "-mem-path option unsupported\n");

            exit(1);

#endif

        } else {

            if (xen_enabled()) {

                xen_ram_alloc(new_block->offset, size, mr);

            } else if (kvm_enabled()) {

                /* some s390/kvm configurations have special constraints */

                new_block->host = kvm_vmalloc(size);

            } else {

                new_block->host = qemu_vmalloc(size);

            }

            qemu_madvise(new_block->host, size, QEMU_MADV_MERGEABLE);

        }

    }

    new_block->length = size;



    QLIST_INSERT_HEAD(&ram_list.blocks, new_block, next);



    ram_list.phys_dirty = g_realloc(ram_list.phys_dirty,

                                       last_ram_offset() >> TARGET_PAGE_BITS);



    cpu_physical_memory_set_dirty_range(new_block->offset, size, 0xff);



    if (kvm_enabled())

        kvm_setup_guest_memory(new_block->host, size);



    return new_block->offset;

}