static int ram_save_block(QEMUFile *f, bool last_stage)

{

    RAMBlock *block = last_seen_block;

    ram_addr_t offset = last_offset;

    bool complete_round = false;

    int bytes_sent = 0;

    MemoryRegion *mr;

    ram_addr_t current_addr;



    if (!block)

        block = QTAILQ_FIRST(&ram_list.blocks);



    while (true) {

        mr = block->mr;

        offset = migration_bitmap_find_and_reset_dirty(mr, offset);

        if (complete_round && block == last_seen_block &&

            offset >= last_offset) {

            break;

        }

        if (offset >= block->length) {

            offset = 0;

            block = QTAILQ_NEXT(block, next);

            if (!block) {

                block = QTAILQ_FIRST(&ram_list.blocks);

                complete_round = true;

                ram_bulk_stage = false;

            }

        } else {

            int ret;

            uint8_t *p;

            int cont = (block == last_sent_block) ?

                RAM_SAVE_FLAG_CONTINUE : 0;



            p = memory_region_get_ram_ptr(mr) + offset;



            /* In doubt sent page as normal */

            bytes_sent = -1;

            ret = ram_control_save_page(f, block->offset,

                               offset, TARGET_PAGE_SIZE, &bytes_sent);



            if (ret != RAM_SAVE_CONTROL_NOT_SUPP) {

                if (ret != RAM_SAVE_CONTROL_DELAYED) {

                    if (bytes_sent > 0) {

                        acct_info.norm_pages++;

                    } else if (bytes_sent == 0) {

                        acct_info.dup_pages++;

                    }

                }

            } else if (is_zero_range(p, TARGET_PAGE_SIZE)) {

                acct_info.dup_pages++;

                bytes_sent = save_block_hdr(f, block, offset, cont,

                                            RAM_SAVE_FLAG_COMPRESS);

                qemu_put_byte(f, 0);

                bytes_sent++;

            } else if (!ram_bulk_stage && migrate_use_xbzrle()) {

                current_addr = block->offset + offset;

                bytes_sent = save_xbzrle_page(f, p, current_addr, block,

                                              offset, cont, last_stage);

                if (!last_stage) {

                    p = get_cached_data(XBZRLE.cache, current_addr);

                }

            }



            /* XBZRLE overflow or normal page */

            if (bytes_sent == -1) {

                bytes_sent = save_block_hdr(f, block, offset, cont, RAM_SAVE_FLAG_PAGE);

                qemu_put_buffer_async(f, p, TARGET_PAGE_SIZE);

                bytes_sent += TARGET_PAGE_SIZE;

                acct_info.norm_pages++;

            }



            /* if page is unmodified, continue to the next */

            if (bytes_sent > 0) {

                last_sent_block = block;

                break;

            }

        }

    }

    last_seen_block = block;

    last_offset = offset;



    return bytes_sent;

}
