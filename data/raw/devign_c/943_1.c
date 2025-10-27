static int save_xbzrle_page(RAMState *rs, uint8_t **current_data,

                            ram_addr_t current_addr, RAMBlock *block,

                            ram_addr_t offset, bool last_stage)

{

    int encoded_len = 0, bytes_xbzrle;

    uint8_t *prev_cached_page;



    if (!cache_is_cached(XBZRLE.cache, current_addr, rs->bitmap_sync_count)) {

        rs->xbzrle_cache_miss++;

        if (!last_stage) {

            if (cache_insert(XBZRLE.cache, current_addr, *current_data,

                             rs->bitmap_sync_count) == -1) {

                return -1;

            } else {

                /* update *current_data when the page has been

                   inserted into cache */

                *current_data = get_cached_data(XBZRLE.cache, current_addr);

            }

        }

        return -1;

    }



    prev_cached_page = get_cached_data(XBZRLE.cache, current_addr);



    /* save current buffer into memory */

    memcpy(XBZRLE.current_buf, *current_data, TARGET_PAGE_SIZE);



    /* XBZRLE encoding (if there is no overflow) */

    encoded_len = xbzrle_encode_buffer(prev_cached_page, XBZRLE.current_buf,

                                       TARGET_PAGE_SIZE, XBZRLE.encoded_buf,

                                       TARGET_PAGE_SIZE);

    if (encoded_len == 0) {

        trace_save_xbzrle_page_skipping();

        return 0;

    } else if (encoded_len == -1) {

        trace_save_xbzrle_page_overflow();

        rs->xbzrle_overflows++;

        /* update data in the cache */

        if (!last_stage) {

            memcpy(prev_cached_page, *current_data, TARGET_PAGE_SIZE);

            *current_data = prev_cached_page;

        }

        return -1;

    }



    /* we need to update the data in the cache, in order to get the same data */

    if (!last_stage) {

        memcpy(prev_cached_page, XBZRLE.current_buf, TARGET_PAGE_SIZE);

    }



    /* Send XBZRLE based compressed page */

    bytes_xbzrle = save_page_header(rs, block,

                                    offset | RAM_SAVE_FLAG_XBZRLE);

    qemu_put_byte(rs->f, ENCODING_FLAG_XBZRLE);

    qemu_put_be16(rs->f, encoded_len);

    qemu_put_buffer(rs->f, XBZRLE.encoded_buf, encoded_len);

    bytes_xbzrle += encoded_len + 1 + 2;

    rs->xbzrle_pages++;

    rs->xbzrle_bytes += bytes_xbzrle;

    rs->bytes_transferred += bytes_xbzrle;



    return 1;

}
