static void coroutine_fn bdrv_get_block_status_above_co_entry(void *opaque)

{

    BdrvCoGetBlockStatusData *data = opaque;



    data->ret = bdrv_co_get_block_status_above(data->bs, data->base,

                                               data->sector_num,

                                               data->nb_sectors,

                                               data->pnum);

    data->done = true;

}
