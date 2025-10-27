static BlockJob *find_block_job(const char *device)

{

    BlockDriverState *bs;



    bs = bdrv_find(device);

    if (!bs || !bs->job) {

        return NULL;

    }

    return bs->job;

}
