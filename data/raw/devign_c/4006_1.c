static BlockStats *bdrv_query_bds_stats(const BlockDriverState *bs,

                                 bool query_backing)

{

    BlockStats *s = NULL;



    s = g_malloc0(sizeof(*s));

    s->stats = g_malloc0(sizeof(*s->stats));



    if (!bs) {

        return s;

    }



    if (bdrv_get_node_name(bs)[0]) {

        s->has_node_name = true;

        s->node_name = g_strdup(bdrv_get_node_name(bs));

    }



    s->stats->wr_highest_offset = stat64_get(&bs->wr_highest_offset);



    if (bs->file) {

        s->has_parent = true;

        s->parent = bdrv_query_bds_stats(bs->file->bs, query_backing);

    }



    if (query_backing && bs->backing) {

        s->has_backing = true;

        s->backing = bdrv_query_bds_stats(bs->backing->bs, query_backing);

    }



    return s;

}
