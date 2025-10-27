static ram_addr_t ram_block_add(RAMBlock *new_block, Error **errp)

{

    RAMBlock *block;

    RAMBlock *last_block = NULL;

    ram_addr_t old_ram_size, new_ram_size;



    old_ram_size = last_ram_offset() >> TARGET_PAGE_BITS;



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


            memory_try_enable_merging(new_block->host, new_block->max_length);










    /* Keep the list sorted from biggest to smallest block.  Unlike QTAILQ,

     * QLIST (which has an RCU-friendly variant) does not have insertion at

     * tail, so save the last element in last_block.

     */

    QLIST_FOREACH_RCU(block, &ram_list.blocks, next) {

        last_block = block;

        if (block->max_length < new_block->max_length) {

            break;



    if (block) {

        QLIST_INSERT_BEFORE_RCU(block, new_block, next);

    } else if (last_block) {

        QLIST_INSERT_AFTER_RCU(last_block, new_block, next);

    } else { /* list is empty */

        QLIST_INSERT_HEAD_RCU(&ram_list.blocks, new_block, next);


    ram_list.mru_block = NULL;



    /* Write list before version */

    smp_wmb();

    ram_list.version++;

    qemu_mutex_unlock_ramlist();



    new_ram_size = last_ram_offset() >> TARGET_PAGE_BITS;




        int i;



        /* ram_list.dirty_memory[] is protected by the iothread lock.  */

        for (i = 0; i < DIRTY_MEMORY_NUM; i++) {

            ram_list.dirty_memory[i] =

                bitmap_zero_extend(ram_list.dirty_memory[i],

                                   old_ram_size, new_ram_size);



    cpu_physical_memory_set_dirty_range(new_block->offset,

                                        new_block->used_length,

                                        DIRTY_CLIENTS_ALL);



    if (new_block->host) {

        qemu_ram_setup_dump(new_block->host, new_block->max_length);

        qemu_madvise(new_block->host, new_block->max_length, QEMU_MADV_HUGEPAGE);

        qemu_madvise(new_block->host, new_block->max_length, QEMU_MADV_DONTFORK);

        if (kvm_enabled()) {

            kvm_setup_guest_memory(new_block->host, new_block->max_length);





    return new_block->offset;
