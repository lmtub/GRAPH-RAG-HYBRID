static inline uint64_t vmdk_find_offset_in_cluster(VmdkExtent *extent,

                                                   int64_t offset)

{

    uint64_t offset_in_cluster, extent_begin_offset, extent_relative_offset;

    uint64_t cluster_size = extent->cluster_sectors * BDRV_SECTOR_SIZE;



    extent_begin_offset =

        (extent->end_sector - extent->sectors) * BDRV_SECTOR_SIZE;

    extent_relative_offset = offset - extent_begin_offset;

    offset_in_cluster = extent_relative_offset % cluster_size;



    return offset_in_cluster;

}
