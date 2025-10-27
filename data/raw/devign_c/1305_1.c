BlockDeviceInfoList *qmp_query_named_block_nodes(Error **errp)

{

    return bdrv_named_nodes_list();

}
