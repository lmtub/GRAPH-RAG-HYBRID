void hmp_info_block_jobs(Monitor *mon, const QDict *qdict)
{
    BlockJobInfoList *list;
    Error *err = NULL;
    list = qmp_query_block_jobs(&err);
    assert(!err);
    if (!list) {
        monitor_printf(mon, "No active jobs\n");
        return;
    }
    while (list) {
        if (strcmp(list->value->type, "stream") == 0) {
            monitor_printf(mon, "Streaming device %s: Completed %" PRId64
                           " of %" PRId64 " bytes, speed limit %" PRId64
                           " bytes/s\n",
                           list->value->device,
                           list->value->offset,
                           list->value->len,
                           list->value->speed);
        } else {
            monitor_printf(mon, "Type %s, device %s: Completed %" PRId64
                           " of %" PRId64 " bytes, speed limit %" PRId64
                           " bytes/s\n",
                           list->value->type,
                           list->value->device,
                           list->value->offset,
                           list->value->len,
                           list->value->speed);
        }
        list = list->next;
    }
}