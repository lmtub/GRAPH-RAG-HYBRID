static int ram_save_block(QEMUFile *f, bool last_stage)

{

    RAMBlock *block = last_seen_block;

    ram_addr_t offset = last_offset;

    int bytes_sent = -1;

    MemoryRegion *mr;

    ram_addr_t current_addr;



    if (!block)

        block = QTAILQ_FIRST(&ram_list.blocks);



    do {

        mr = block->mr;

        if (migration_bitmap_test_and_reset_dirty(mr, offset)) {

            uint8_t *p;

            int cont = (block == last_sent_block) ?

                RAM_SAVE_FLAG_CONTINUE : 0;



            p = memory_region_get_ram_ptr(mr) + offset;



            if (is_dup_page(p)) {

                acct_info.dup_pages++;

                save_block_hdr(f, block, offset, cont, RAM_SAVE_FLAG_COMPRESS);

                qemu_put_byte(f, *p);

                bytes_sent = 1;

            } else if (migrate_use_xbzrle()) {

                current_addr = block->offset + offset;

                bytes_sent = save_xbzrle_page(f, p, current_addr, block,

                                              offset, cont, last_stage);

                if (!last_stage) {

                    p = get_cached_data(XBZRLE.cache, current_addr);

                }

            }



            /* either we didn't send yet (we may have had XBZRLE overflow) */

            if (bytes_sent == -1) {

                save_block_hdr(f, block, offset, cont, RAM_SAVE_FLAG_PAGE);

                qemu_put_buffer(f, p, TARGET_PAGE_SIZE);

                bytes_sent = TARGET_PAGE_SIZE;

                acct_info.norm_pages++;

            }



            /* if page is unmodified, continue to the next */

            if (bytes_sent != 0) {

                last_sent_block = block;

                break;

            }

        }



        offset += TARGET_PAGE_SIZE;

        if (offset >= block->length) {

            offset = 0;

            block = QTAILQ_NEXT(block, next);

            if (!block)

                block = QTAILQ_FIRST(&ram_list.blocks);

        }

    } while (block != last_seen_block || offset != last_offset);



    last_seen_block = block;

    last_offset = offset;



    return bytes_sent;

}
