static int grow_refcount_table(BlockDriverState *bs, int min_size)

{

    BDRVQcowState *s = bs->opaque;

    int new_table_size, new_table_size2, refcount_table_clusters, i, ret;

    uint64_t *new_table;

    int64_t table_offset;

    uint8_t data[12];

    int old_table_size;

    int64_t old_table_offset;



    if (min_size <= s->refcount_table_size)

        return 0;

    /* compute new table size */

    new_table_size = next_refcount_table_size(s, min_size);

#ifdef DEBUG_ALLOC2

    printf("grow_refcount_table from %d to %d\n",

           s->refcount_table_size,

           new_table_size);

#endif

    new_table_size2 = new_table_size * sizeof(uint64_t);

    new_table = qemu_mallocz(new_table_size2);

    memcpy(new_table, s->refcount_table,

           s->refcount_table_size * sizeof(uint64_t));

    for(i = 0; i < s->refcount_table_size; i++)

        cpu_to_be64s(&new_table[i]);

    /* Note: we cannot update the refcount now to avoid recursion */

    table_offset = alloc_clusters_noref(bs, new_table_size2);

    ret = bdrv_pwrite(s->hd, table_offset, new_table, new_table_size2);

    if (ret != new_table_size2)

        goto fail;

    for(i = 0; i < s->refcount_table_size; i++)

        be64_to_cpus(&new_table[i]);



    cpu_to_be64w((uint64_t*)data, table_offset);

    cpu_to_be32w((uint32_t*)(data + 8), refcount_table_clusters);

    ret = bdrv_pwrite(s->hd, offsetof(QCowHeader, refcount_table_offset),

                    data, sizeof(data));

    if (ret != sizeof(data)) {

        goto fail;

    }



    qemu_free(s->refcount_table);

    old_table_offset = s->refcount_table_offset;

    old_table_size = s->refcount_table_size;

    s->refcount_table = new_table;

    s->refcount_table_size = new_table_size;

    s->refcount_table_offset = table_offset;



    update_refcount(bs, table_offset, new_table_size2, 1);

    qcow2_free_clusters(bs, old_table_offset, old_table_size * sizeof(uint64_t));

    return 0;

 fail:

    qemu_free(new_table);

    return ret < 0 ? ret : -EIO;

}
