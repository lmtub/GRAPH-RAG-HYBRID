static int vdi_co_read(BlockDriverState *bs,

        int64_t sector_num, uint8_t *buf, int nb_sectors)

{

    BDRVVdiState *s = bs->opaque;

    uint32_t bmap_entry;

    uint32_t block_index;

    uint32_t sector_in_block;

    uint32_t n_sectors;

    int ret;



    logout("\n");



restart:

    block_index = sector_num / s->block_sectors;

    sector_in_block = sector_num % s->block_sectors;

    n_sectors = s->block_sectors - sector_in_block;

    if (n_sectors > nb_sectors) {

        n_sectors = nb_sectors;

    }



    logout("will read %u sectors starting at sector %" PRIu64 "\n",

           n_sectors, sector_num);



    /* prepare next AIO request */

    bmap_entry = le32_to_cpu(s->bmap[block_index]);

    if (!VDI_IS_ALLOCATED(bmap_entry)) {

        /* Block not allocated, return zeros, no need to wait. */

        memset(buf, 0, n_sectors * SECTOR_SIZE);

        ret = 0;

    } else {

        uint64_t offset = s->header.offset_data / SECTOR_SIZE +

                          (uint64_t)bmap_entry * s->block_sectors +

                          sector_in_block;

        ret = bdrv_read(bs->file, offset, buf, n_sectors);

    }

    logout("%u sectors read\n", n_sectors);



    nb_sectors -= n_sectors;

    sector_num += n_sectors;

    buf += n_sectors * SECTOR_SIZE;



    if (ret >= 0 && nb_sectors > 0) {

        goto restart;

    }



    return ret;

}
