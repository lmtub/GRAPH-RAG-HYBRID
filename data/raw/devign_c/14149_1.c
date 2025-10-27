void qemu_get_guest_simple_memory_mapping(MemoryMappingList *list)

{

    RAMBlock *block;



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        create_new_memory_mapping(list, block->offset, 0, block->length);

    }

}
