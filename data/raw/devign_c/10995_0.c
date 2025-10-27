static void iscsi_allocationmap_clear(IscsiLun *iscsilun, int64_t sector_num,

                                      int nb_sectors)

{

    int64_t cluster_num, nb_clusters;

    if (iscsilun->allocationmap == NULL) {

        return;

    }

    cluster_num = DIV_ROUND_UP(sector_num, iscsilun->cluster_sectors);

    nb_clusters = (sector_num + nb_sectors) / iscsilun->cluster_sectors

                  - cluster_num;

    if (nb_clusters > 0) {

        bitmap_clear(iscsilun->allocationmap, cluster_num, nb_clusters);

    }

}
