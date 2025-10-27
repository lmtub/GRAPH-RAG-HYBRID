static void alloc_aio_bitmap(BlkMigDevState *bmds)

{

    BlockDriverState *bs = bmds->bs;

    int64_t bitmap_size;



    bitmap_size = bdrv_nb_sectors(bs) + BDRV_SECTORS_PER_DIRTY_CHUNK * 8 - 1;

    bitmap_size /= BDRV_SECTORS_PER_DIRTY_CHUNK * 8;



    bmds->aio_bitmap = g_malloc0(bitmap_size);

}
