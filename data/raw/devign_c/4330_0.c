static int qcow_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVQcowState *s = bs->opaque;

    int len, i, shift, ret;

    QCowHeader header;

    uint64_t ext_end;



    ret = bdrv_file_open(&s->hd, filename, flags);

    if (ret < 0)

        return ret;

    if (bdrv_pread(s->hd, 0, &header, sizeof(header)) != sizeof(header))

        goto fail;

    be32_to_cpus(&header.magic);

    be32_to_cpus(&header.version);

    be64_to_cpus(&header.backing_file_offset);

    be32_to_cpus(&header.backing_file_size);

    be64_to_cpus(&header.size);

    be32_to_cpus(&header.cluster_bits);

    be32_to_cpus(&header.crypt_method);

    be64_to_cpus(&header.l1_table_offset);

    be32_to_cpus(&header.l1_size);

    be64_to_cpus(&header.refcount_table_offset);

    be32_to_cpus(&header.refcount_table_clusters);

    be64_to_cpus(&header.snapshots_offset);

    be32_to_cpus(&header.nb_snapshots);



    if (header.magic != QCOW_MAGIC || header.version != QCOW_VERSION)

        goto fail;

    if (header.size <= 1 ||

        header.cluster_bits < MIN_CLUSTER_BITS ||

        header.cluster_bits > MAX_CLUSTER_BITS)

        goto fail;

    if (header.crypt_method > QCOW_CRYPT_AES)

        goto fail;

    s->crypt_method_header = header.crypt_method;

    if (s->crypt_method_header)

        bs->encrypted = 1;

    s->cluster_bits = header.cluster_bits;

    s->cluster_size = 1 << s->cluster_bits;

    s->cluster_sectors = 1 << (s->cluster_bits - 9);

    s->l2_bits = s->cluster_bits - 3; /* L2 is always one cluster */

    s->l2_size = 1 << s->l2_bits;

    bs->total_sectors = header.size / 512;

    s->csize_shift = (62 - (s->cluster_bits - 8));

    s->csize_mask = (1 << (s->cluster_bits - 8)) - 1;

    s->cluster_offset_mask = (1LL << s->csize_shift) - 1;

    s->refcount_table_offset = header.refcount_table_offset;

    s->refcount_table_size =

        header.refcount_table_clusters << (s->cluster_bits - 3);



    s->snapshots_offset = header.snapshots_offset;

    s->nb_snapshots = header.nb_snapshots;



    /* read the level 1 table */

    s->l1_size = header.l1_size;

    shift = s->cluster_bits + s->l2_bits;

    s->l1_vm_state_index = (header.size + (1LL << shift) - 1) >> shift;

    /* the L1 table must contain at least enough entries to put

       header.size bytes */

    if (s->l1_size < s->l1_vm_state_index)

        goto fail;

    s->l1_table_offset = header.l1_table_offset;

    s->l1_table = qemu_mallocz(

        align_offset(s->l1_size * sizeof(uint64_t), 512));

    if (bdrv_pread(s->hd, s->l1_table_offset, s->l1_table, s->l1_size * sizeof(uint64_t)) !=

        s->l1_size * sizeof(uint64_t))

        goto fail;

    for(i = 0;i < s->l1_size; i++) {

        be64_to_cpus(&s->l1_table[i]);

    }

    /* alloc L2 cache */

    s->l2_cache = qemu_malloc(s->l2_size * L2_CACHE_SIZE * sizeof(uint64_t));

    s->cluster_cache = qemu_malloc(s->cluster_size);

    /* one more sector for decompressed data alignment */

    s->cluster_data = qemu_malloc(QCOW_MAX_CRYPT_CLUSTERS * s->cluster_size

                                  + 512);

    s->cluster_cache_offset = -1;



    if (qcow2_refcount_init(bs) < 0)

        goto fail;



    LIST_INIT(&s->cluster_allocs);



    /* read qcow2 extensions */

    if (header.backing_file_offset)

        ext_end = header.backing_file_offset;

    else

        ext_end = s->cluster_size;

    if (qcow_read_extensions(bs, sizeof(header), ext_end))

        goto fail;



    /* read the backing file name */

    if (header.backing_file_offset != 0) {

        len = header.backing_file_size;

        if (len > 1023)

            len = 1023;

        if (bdrv_pread(s->hd, header.backing_file_offset, bs->backing_file, len) != len)

            goto fail;

        bs->backing_file[len] = '\0';

    }

    if (qcow2_read_snapshots(bs) < 0)

        goto fail;



#ifdef DEBUG_ALLOC

    qcow2_check_refcounts(bs);

#endif

    return 0;



 fail:

    qcow2_free_snapshots(bs);

    qcow2_refcount_close(bs);

    qemu_free(s->l1_table);

    qemu_free(s->l2_cache);

    qemu_free(s->cluster_cache);

    qemu_free(s->cluster_data);

    bdrv_delete(s->hd);

    return -1;

}
