void qemu_ram_free_from_ptr(ram_addr_t addr)

{

    RAMBlock *block;



    /* This assumes the iothread lock is taken here too.  */

    qemu_mutex_lock_ramlist();

    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (addr == block->offset) {

            QTAILQ_REMOVE(&ram_list.blocks, block, next);

            ram_list.mru_block = NULL;

            ram_list.version++;

            g_free(block);

            break;

        }

    }

    qemu_mutex_unlock_ramlist();

}
