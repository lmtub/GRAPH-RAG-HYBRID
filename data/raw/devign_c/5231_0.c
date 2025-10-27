static int ram_save_page(QEMUFile *f, PageSearchStatus *pss,

                         bool last_stage, uint64_t *bytes_transferred)

{

    int pages = -1;

    uint64_t bytes_xmit;

    ram_addr_t current_addr;

    uint8_t *p;

    int ret;

    bool send_async = true;

    RAMBlock *block = pss->block;

    ram_addr_t offset = pss->offset;



    p = block->host + offset;



    /* In doubt sent page as normal */

    bytes_xmit = 0;

    ret = ram_control_save_page(f, block->offset,

                           offset, TARGET_PAGE_SIZE, &bytes_xmit);

    if (bytes_xmit) {

        *bytes_transferred += bytes_xmit;

        pages = 1;

    }



    XBZRLE_cache_lock();



    current_addr = block->offset + offset;



    if (block == last_sent_block) {

        offset |= RAM_SAVE_FLAG_CONTINUE;

    }

    if (ret != RAM_SAVE_CONTROL_NOT_SUPP) {

        if (ret != RAM_SAVE_CONTROL_DELAYED) {

            if (bytes_xmit > 0) {

                acct_info.norm_pages++;

            } else if (bytes_xmit == 0) {

                acct_info.dup_pages++;

            }

        }

    } else {

        pages = save_zero_page(f, block, offset, p, bytes_transferred);

        if (pages > 0) {

            /* Must let xbzrle know, otherwise a previous (now 0'd) cached

             * page would be stale

             */

            xbzrle_cache_zero_page(current_addr);

        } else if (!ram_bulk_stage && migrate_use_xbzrle()) {

            pages = save_xbzrle_page(f, &p, current_addr, block,

                                     offset, last_stage, bytes_transferred);

            if (!last_stage) {

                /* Can't send this cached data async, since the cache page

                 * might get updated before it gets to the wire

                 */

                send_async = false;

            }

        }

    }



    /* XBZRLE overflow or normal page */

    if (pages == -1) {

        *bytes_transferred += save_page_header(f, block,

                                               offset | RAM_SAVE_FLAG_PAGE);

        if (send_async) {

            qemu_put_buffer_async(f, p, TARGET_PAGE_SIZE);

        } else {

            qemu_put_buffer(f, p, TARGET_PAGE_SIZE);

        }

        *bytes_transferred += TARGET_PAGE_SIZE;

        pages = 1;

        acct_info.norm_pages++;

    }



    XBZRLE_cache_unlock();



    return pages;

}
