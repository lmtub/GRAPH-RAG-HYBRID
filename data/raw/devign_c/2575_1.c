static int alloc_cluster_link_l2(BlockDriverState *bs, uint64_t cluster_offset,

        QCowL2Meta *m)

{

    BDRVQcowState *s = bs->opaque;

    int i, j = 0, l2_index, ret;

    uint64_t *old_cluster, start_sect, l2_offset, *l2_table;



    if (m->nb_clusters == 0)

        return 0;



    old_cluster = qemu_malloc(m->nb_clusters * sizeof(uint64_t));



    /* copy content of unmodified sectors */

    start_sect = (m->offset & ~(s->cluster_size - 1)) >> 9;

    if (m->n_start) {

        ret = copy_sectors(bs, start_sect, cluster_offset, 0, m->n_start);

        if (ret < 0)

            goto err;

    }



    if (m->nb_available & (s->cluster_sectors - 1)) {

        uint64_t end = m->nb_available & ~(uint64_t)(s->cluster_sectors - 1);

        ret = copy_sectors(bs, start_sect + end, cluster_offset + (end << 9),

                m->nb_available - end, s->cluster_sectors);

        if (ret < 0)

            goto err;

    }



    ret = -EIO;

    /* update L2 table */

    if (!get_cluster_table(bs, m->offset, &l2_table, &l2_offset, &l2_index))

        goto err;



    for (i = 0; i < m->nb_clusters; i++) {

        if(l2_table[l2_index + i] != 0)

            old_cluster[j++] = l2_table[l2_index + i];



        l2_table[l2_index + i] = cpu_to_be64((cluster_offset +

                    (i << s->cluster_bits)) | QCOW_OFLAG_COPIED);

     }



    if (bdrv_pwrite(s->hd, l2_offset + l2_index * sizeof(uint64_t),

                l2_table + l2_index, m->nb_clusters * sizeof(uint64_t)) !=

            m->nb_clusters * sizeof(uint64_t))

        goto err;



    for (i = 0; i < j; i++)

        free_any_clusters(bs, old_cluster[i], 1);



    ret = 0;

err:

    qemu_free(old_cluster);

    return ret;

 }
