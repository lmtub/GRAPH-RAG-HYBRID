int qcow2_snapshot_goto(BlockDriverState *bs, const char *snapshot_id)

{

    BDRVQcowState *s = bs->opaque;

    QCowSnapshot *sn;

    int i, snapshot_index, l1_size2;



    snapshot_index = find_snapshot_by_id_or_name(bs, snapshot_id);

    if (snapshot_index < 0)

        return -ENOENT;

    sn = &s->snapshots[snapshot_index];



    if (qcow2_update_snapshot_refcount(bs, s->l1_table_offset, s->l1_size, -1) < 0)

        goto fail;



    if (qcow2_grow_l1_table(bs, sn->l1_size) < 0)

        goto fail;



    s->l1_size = sn->l1_size;

    l1_size2 = s->l1_size * sizeof(uint64_t);

    /* copy the snapshot l1 table to the current l1 table */

    if (bdrv_pread(bs->file, sn->l1_table_offset,

                   s->l1_table, l1_size2) != l1_size2)

        goto fail;

    if (bdrv_pwrite(bs->file, s->l1_table_offset,

                    s->l1_table, l1_size2) != l1_size2)

        goto fail;

    for(i = 0;i < s->l1_size; i++) {

        be64_to_cpus(&s->l1_table[i]);

    }



    if (qcow2_update_snapshot_refcount(bs, s->l1_table_offset, s->l1_size, 1) < 0)

        goto fail;



#ifdef DEBUG_ALLOC

    qcow2_check_refcounts(bs);

#endif

    return 0;

 fail:

    return -EIO;

}
