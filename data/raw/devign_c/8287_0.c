ram_addr_t qemu_ram_alloc_from_ptr(DeviceState *dev, const char *name,

                        ram_addr_t size, void *host)

{

    RAMBlock *new_block, *block;



    size = TARGET_PAGE_ALIGN(size);

    new_block = qemu_mallocz(sizeof(*new_block));



    if (dev && dev->parent_bus && dev->parent_bus->info->get_dev_path) {

        char *id = dev->parent_bus->info->get_dev_path(dev);

        if (id) {

            snprintf(new_block->idstr, sizeof(new_block->idstr), "%s/", id);

            qemu_free(id);

        }

    }

    pstrcat(new_block->idstr, sizeof(new_block->idstr), name);



    QLIST_FOREACH(block, &ram_list.blocks, next) {

        if (!strcmp(block->idstr, new_block->idstr)) {

            fprintf(stderr, "RAMBlock \"%s\" already registered, abort!\n",

                    new_block->idstr);

            abort();

        }

    }



    new_block->host = host;



    new_block->offset = find_ram_offset(size);

    new_block->length = size;



    QLIST_INSERT_HEAD(&ram_list.blocks, new_block, next);



    ram_list.phys_dirty = qemu_realloc(ram_list.phys_dirty,

                                       last_ram_offset() >> TARGET_PAGE_BITS);

    memset(ram_list.phys_dirty + (new_block->offset >> TARGET_PAGE_BITS),

           0xff, size >> TARGET_PAGE_BITS);



    if (kvm_enabled())

        kvm_setup_guest_memory(new_block->host, size);



    return new_block->offset;

}
