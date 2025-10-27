static ram_addr_t ram_block_add(RAMBlock *new_block, Error **errp)

{

    RAMBlock *block;

    ram_addr_t old_ram_size, new_ram_size;



    old_ram_size = last_ram_offset() >> TARGET_PAGE_BITS;



    /* This assumes the iothread lock is taken here too.  */

    qemu_mutex_lock_ramlist();

    new_block->offset = find_ram_offset(new_block->max_length);



    if (!new_block->host) {

        if (xen_enabled()) {

            xen_ram_alloc(new_block->offset, new_block->max_length,

                          new_block->mr);

        } else {

            new_block->host = phys_mem_alloc(new_block->max_length,

                                             &new_block->mr->align);

            if (!new_block->host) {

                error_setg_errno(errp, errno,

                                 "cannot set up guest memory '%s'",

                                 memory_region_name(new_block->mr));

                qemu_mutex_unlock_ramlist();

                return -1;

            }

            memory_try_enable_merging(new_block->host, new_block->max_length);

        }

    }



    /* Keep the list sorted from biggest to smallest block.  */

    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (block->max_length < new_block->max_length) {

            break;

        }

    }

    if (block) {

        QTAILQ_INSERT_BEFORE(block, new_block, next);

    } else {

        QTAILQ_INSERT_TAIL(&ram_list.blocks, new_block, next);

    }

    ram_list.mru_block = NULL;



    ram_list.version++;

    qemu_mutex_unlock_ramlist();



    new_ram_size = last_ram_offset() >> TARGET_PAGE_BITS;



    if (new_ram_size > old_ram_size) {

        int i;

        for (i = 0; i < DIRTY_MEMORY_NUM; i++) {

            ram_list.dirty_memory[i] =

                bitmap_zero_extend(ram_list.dirty_memory[i],

                                   old_ram_size, new_ram_size);

       }

    }

    cpu_physical_memory_set_dirty_range(new_block->offset,

                                        new_block->used_length);



    qemu_ram_setup_dump(new_block->host, new_block->max_length);

    qemu_madvise(new_block->host, new_block->max_length, QEMU_MADV_HUGEPAGE);

    qemu_madvise(new_block->host, new_block->max_length, QEMU_MADV_DONTFORK);



    if (kvm_enabled()) {

        kvm_setup_guest_memory(new_block->host, new_block->max_length);

    }



    return new_block->offset;

}
