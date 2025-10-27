BlockInfoList *qmp_query_block(Error **errp)

{

    BlockInfoList *head = NULL, **p_next = &head;

    BlockBackend *blk;

    Error *local_err = NULL;



    for (blk = blk_next(NULL); blk; blk = blk_next(blk)) {

        BlockInfoList *info = g_malloc0(sizeof(*info));

        bdrv_query_info(blk, &info->value, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            goto err;

        }



        *p_next = info;

        p_next = &info->next;

    }



    return head;



 err:

    qapi_free_BlockInfoList(head);

    return NULL;

}
