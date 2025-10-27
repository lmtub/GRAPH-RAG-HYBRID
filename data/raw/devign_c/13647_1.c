static RAMBlock *unqueue_page(RAMState *rs, ram_addr_t *offset,

                              ram_addr_t *ram_addr_abs)

{

    RAMBlock *block = NULL;



    qemu_mutex_lock(&rs->src_page_req_mutex);

    if (!QSIMPLEQ_EMPTY(&rs->src_page_requests)) {

        struct RAMSrcPageRequest *entry =

                                QSIMPLEQ_FIRST(&rs->src_page_requests);

        block = entry->rb;

        *offset = entry->offset;

        *ram_addr_abs = (entry->offset + entry->rb->offset) &

                        TARGET_PAGE_MASK;



        if (entry->len > TARGET_PAGE_SIZE) {

            entry->len -= TARGET_PAGE_SIZE;

            entry->offset += TARGET_PAGE_SIZE;

        } else {

            memory_region_unref(block->mr);

            QSIMPLEQ_REMOVE_HEAD(&rs->src_page_requests, next_req);

            g_free(entry);

        }

    }

    qemu_mutex_unlock(&rs->src_page_req_mutex);



    return block;

}
