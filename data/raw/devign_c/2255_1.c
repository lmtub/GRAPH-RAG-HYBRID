static int check_refcounts_l1(BlockDriverState *bs,
                              uint16_t *refcount_table,
                              int refcount_table_size,
                              int64_t l1_table_offset, int l1_size,
                              int check_copied)
{
    BDRVQcowState *s = bs->opaque;
    uint64_t *l1_table, l2_offset, l1_size2;
    int i, refcount, ret;
    int errors = 0;
    l1_size2 = l1_size * sizeof(uint64_t);
    /* Mark L1 table as used */
    errors += inc_refcounts(bs, refcount_table, refcount_table_size,
                  l1_table_offset, l1_size2);
    /* Read L1 table entries from disk */
    l1_table = qemu_malloc(l1_size2);
    if (bdrv_pread(s->hd, l1_table_offset,
                   l1_table, l1_size2) != l1_size2)
        goto fail;
    for(i = 0;i < l1_size; i++)
        be64_to_cpus(&l1_table[i]);
    /* Do the actual checks */
    for(i = 0; i < l1_size; i++) {
        l2_offset = l1_table[i];
        if (l2_offset) {
            /* QCOW_OFLAG_COPIED must be set iff refcount == 1 */
            if (check_copied) {
                refcount = get_refcount(bs, (l2_offset & ~QCOW_OFLAG_COPIED)
                    >> s->cluster_bits);
                if ((refcount == 1) != ((l2_offset & QCOW_OFLAG_COPIED) != 0)) {
                    fprintf(stderr, "ERROR OFLAG_COPIED: l2_offset=%" PRIx64
                        " refcount=%d\n", l2_offset, refcount);
            /* Mark L2 table as used */
            l2_offset &= ~QCOW_OFLAG_COPIED;
            errors += inc_refcounts(bs, refcount_table,
                          refcount_table_size,
                          l2_offset,
                          s->cluster_size);
            /* Process and check L2 entries */
            ret = check_refcounts_l2(bs, refcount_table, refcount_table_size,
                l2_offset, check_copied);
            if (ret < 0) {
                goto fail;
            errors += ret;
    qemu_free(l1_table);
    return errors;
fail:
    fprintf(stderr, "ERROR: I/O error in check_refcounts_l1\n");
    qemu_free(l1_table);
    return -EIO;