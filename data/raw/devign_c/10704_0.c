BlockDriverState *bdrv_find_node(const char *node_name)

{

    BlockDriverState *bs;



    assert(node_name);



    QTAILQ_FOREACH(bs, &graph_bdrv_states, node_list) {

        if (!strcmp(node_name, bs->node_name)) {

            return bs;

        }

    }

    return NULL;

}
