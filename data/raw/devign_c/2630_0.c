static void vvfat_refresh_limits(BlockDriverState *bs, Error **errp)

{

    bs->request_alignment = BDRV_SECTOR_SIZE; /* No sub-sector I/O supported */

}
