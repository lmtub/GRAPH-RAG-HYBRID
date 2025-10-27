BlockDeviceInfoList *bdrv_named_nodes_list(void)

{

    BlockDeviceInfoList *list, *entry;

    BlockDriverState *bs;



    list = NULL;

    QTAILQ_FOREACH(bs, &graph_bdrv_states, node_list) {

        entry = g_malloc0(sizeof(*entry));

        entry->value = bdrv_block_device_info(bs);

        entry->next = list;

        list = entry;

    }



    return list;

}
