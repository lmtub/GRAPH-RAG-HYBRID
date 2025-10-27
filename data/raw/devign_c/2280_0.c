static int ram_save_block(QEMUFile *f)

{

    RAMBlock *block = last_block;

    ram_addr_t offset = last_offset;

    int bytes_sent = -1;

    MemoryRegion *mr;



    if (!block)

        block = QLIST_FIRST(&ram_list.blocks);



    do {

        mr = block->mr;

        if (memory_region_get_dirty(mr, offset, TARGET_PAGE_SIZE,

                                    DIRTY_MEMORY_MIGRATION)) {

            uint8_t *p;

            int cont = (block == last_block) ? RAM_SAVE_FLAG_CONTINUE : 0;



            memory_region_reset_dirty(mr, offset, TARGET_PAGE_SIZE,

                                      DIRTY_MEMORY_MIGRATION);



            p = memory_region_get_ram_ptr(mr) + offset;



            if (is_dup_page(p)) {

                save_block_hdr(f, block, offset, cont, RAM_SAVE_FLAG_COMPRESS);

                qemu_put_byte(f, *p);

                bytes_sent = 1;

            } else {

                save_block_hdr(f, block, offset, cont, RAM_SAVE_FLAG_PAGE);

                qemu_put_buffer(f, p, TARGET_PAGE_SIZE);

                bytes_sent = TARGET_PAGE_SIZE;

            }



            break;

        }



        offset += TARGET_PAGE_SIZE;

        if (offset >= block->length) {

            offset = 0;

            block = QLIST_NEXT(block, next);

            if (!block)

                block = QLIST_FIRST(&ram_list.blocks);

        }

    } while (block != last_block || offset != last_offset);



    last_block = block;

    last_offset = offset;



    return bytes_sent;

}
