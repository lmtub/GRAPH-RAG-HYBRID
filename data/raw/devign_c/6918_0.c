static int raw_pwrite_aligned(BlockDriverState *bs, int64_t offset,

                      const uint8_t *buf, int count)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    ret = fd_open(bs);

    if (ret < 0)

        return -errno;



    ret = pwrite(s->fd, buf, count, offset);

    if (ret == count)

        goto label__raw_write__success;



    DEBUG_BLOCK_PRINT("raw_pwrite(%d:%s, %" PRId64 ", %p, %d) [%" PRId64

                      "] write failed %d : %d = %s\n",

                      s->fd, bs->filename, offset, buf, count,

                      bs->total_sectors, ret, errno, strerror(errno));



label__raw_write__success:



    return  (ret < 0) ? -errno : ret;

}
