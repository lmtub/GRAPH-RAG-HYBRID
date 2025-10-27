static int cd_read_sector(IDEState *s, int lba, uint8_t *buf, int sector_size)

{

    int ret;



    switch(sector_size) {

    case 2048:

        block_acct_start(blk_get_stats(s->blk), &s->acct,

                         4 * BDRV_SECTOR_SIZE, BLOCK_ACCT_READ);

        ret = blk_read(s->blk, (int64_t)lba << 2, buf, 4);

        block_acct_done(blk_get_stats(s->blk), &s->acct);

        break;

    case 2352:

        block_acct_start(blk_get_stats(s->blk), &s->acct,

                         4 * BDRV_SECTOR_SIZE, BLOCK_ACCT_READ);

        ret = blk_read(s->blk, (int64_t)lba << 2, buf + 16, 4);

        block_acct_done(blk_get_stats(s->blk), &s->acct);

        if (ret < 0)

            return ret;

        cd_data_to_raw(buf, lba);

        break;

    default:

        ret = -EIO;

        break;

    }

    return ret;

}
