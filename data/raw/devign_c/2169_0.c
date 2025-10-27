void qemu_ram_foreach_block(RAMBlockIterFunc func, void *opaque)

{

    RAMBlock *block;



    rcu_read_lock();

    QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

        func(block->host, block->offset, block->used_length, opaque);

    }

    rcu_read_unlock();

}
