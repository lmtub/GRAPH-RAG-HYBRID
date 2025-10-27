BlockDriverState *bdrv_next_node(BlockDriverState *bs)

{

    if (!bs) {

        return QTAILQ_FIRST(&graph_bdrv_states);

    }

    return QTAILQ_NEXT(bs, node_list);

}
