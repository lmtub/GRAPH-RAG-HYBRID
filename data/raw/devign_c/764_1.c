uint64_t blk_mig_bytes_total(void)

{

    BlkMigDevState *bmds;

    uint64_t sum = 0;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        sum += bmds->total_sectors;

    }

    return sum << BDRV_SECTOR_BITS;

}
