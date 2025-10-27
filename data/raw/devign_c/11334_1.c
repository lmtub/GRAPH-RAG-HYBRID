static coroutine_fn int vmdk_co_write(BlockDriverState *bs, int64_t sector_num,

                                      const uint8_t *buf, int nb_sectors)

{

    int ret;

    BDRVVmdkState *s = bs->opaque;

    qemu_co_mutex_lock(&s->lock);

    ret = vmdk_write(bs, sector_num, buf, nb_sectors);

    qemu_co_mutex_unlock(&s->lock);

    return ret;

}
