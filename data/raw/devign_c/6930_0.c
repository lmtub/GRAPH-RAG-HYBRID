static int qcow2_cache_do_get(BlockDriverState *bs, Qcow2Cache *c,

    uint64_t offset, void **table, bool read_from_disk)

{

    BDRVQcowState *s = bs->opaque;

    int i;

    int ret;

    int lookup_index;

    uint64_t min_lru_counter = UINT64_MAX;

    int min_lru_index = -1;



    trace_qcow2_cache_get(qemu_coroutine_self(), c == s->l2_table_cache,

                          offset, read_from_disk);



    /* Check if the table is already cached */

    i = lookup_index = (offset / s->cluster_size * 4) % c->size;

    do {

        const Qcow2CachedTable *t = &c->entries[i];

        if (t->offset == offset) {

            goto found;

        }

        if (t->ref == 0 && t->lru_counter < min_lru_counter) {

            min_lru_counter = t->lru_counter;

            min_lru_index = i;

        }

        if (++i == c->size) {

            i = 0;

        }

    } while (i != lookup_index);



    if (min_lru_index == -1) {

        /* This can't happen in current synchronous code, but leave the check

         * here as a reminder for whoever starts using AIO with the cache */

        abort();

    }



    /* Cache miss: write a table back and replace it */

    i = min_lru_index;

    trace_qcow2_cache_get_replace_entry(qemu_coroutine_self(),

                                        c == s->l2_table_cache, i);

    if (i < 0) {

        return i;

    }



    ret = qcow2_cache_entry_flush(bs, c, i);

    if (ret < 0) {

        return ret;

    }



    trace_qcow2_cache_get_read(qemu_coroutine_self(),

                               c == s->l2_table_cache, i);

    c->entries[i].offset = 0;

    if (read_from_disk) {

        if (c == s->l2_table_cache) {

            BLKDBG_EVENT(bs->file, BLKDBG_L2_LOAD);

        }



        ret = bdrv_pread(bs->file, offset, qcow2_cache_get_table_addr(bs, c, i),

                         s->cluster_size);

        if (ret < 0) {

            return ret;

        }

    }



    c->entries[i].offset = offset;



    /* And return the right table */

found:

    c->entries[i].ref++;

    *table = qcow2_cache_get_table_addr(bs, c, i);



    trace_qcow2_cache_get_done(qemu_coroutine_self(),

                               c == s->l2_table_cache, i);



    return 0;

}
