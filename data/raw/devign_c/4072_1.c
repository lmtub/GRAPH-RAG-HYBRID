static int qcow_snapshot_create(BlockDriverState *bs,

                                QEMUSnapshotInfo *sn_info)

{

    BDRVQcowState *s = bs->opaque;

    QCowSnapshot *snapshots1, sn1, *sn = &sn1;

    int i, ret;

    uint64_t *l1_table = NULL;



    memset(sn, 0, sizeof(*sn));



    if (sn_info->id_str[0] == '\0') {

        /* compute a new id */

        find_new_snapshot_id(bs, sn_info->id_str, sizeof(sn_info->id_str));

    }



    /* check that the ID is unique */

    if (find_snapshot_by_id(bs, sn_info->id_str) >= 0)

        return -ENOENT;



    sn->id_str = qemu_strdup(sn_info->id_str);

    if (!sn->id_str)

        goto fail;

    sn->name = qemu_strdup(sn_info->name);

    if (!sn->name)

        goto fail;

    sn->vm_state_size = sn_info->vm_state_size;

    sn->date_sec = sn_info->date_sec;

    sn->date_nsec = sn_info->date_nsec;

    sn->vm_clock_nsec = sn_info->vm_clock_nsec;



    ret = update_snapshot_refcount(bs, s->l1_table_offset, s->l1_size, 1);

    if (ret < 0)

        goto fail;



    /* create the L1 table of the snapshot */

    sn->l1_table_offset = alloc_clusters(bs, s->l1_size * sizeof(uint64_t));

    sn->l1_size = s->l1_size;



    l1_table = qemu_malloc(s->l1_size * sizeof(uint64_t));

    if (!l1_table)

        goto fail;

    for(i = 0; i < s->l1_size; i++) {

        l1_table[i] = cpu_to_be64(s->l1_table[i]);

    }

    if (bdrv_pwrite(s->hd, sn->l1_table_offset,

                    l1_table, s->l1_size * sizeof(uint64_t)) !=

        (s->l1_size * sizeof(uint64_t)))

        goto fail;

    qemu_free(l1_table);

    l1_table = NULL;



    snapshots1 = qemu_malloc((s->nb_snapshots + 1) * sizeof(QCowSnapshot));

    if (!snapshots1)

        goto fail;

    memcpy(snapshots1, s->snapshots, s->nb_snapshots * sizeof(QCowSnapshot));


    s->snapshots = snapshots1;

    s->snapshots[s->nb_snapshots++] = *sn;



    if (qcow_write_snapshots(bs) < 0)

        goto fail;

#ifdef DEBUG_ALLOC

    check_refcounts(bs);

#endif

    return 0;

 fail:

    qemu_free(sn->name);

    qemu_free(l1_table);

    return -1;

}