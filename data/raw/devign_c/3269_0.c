static void qlist_destroy_obj(QObject *obj)

{

    QList *qlist;

    QListEntry *entry, *next_entry;



    assert(obj != NULL);

    qlist = qobject_to_qlist(obj);



    QTAILQ_FOREACH_SAFE(entry, &qlist->head, next, next_entry) {

        QTAILQ_REMOVE(&qlist->head, entry, next);

        qobject_decref(entry->value);

        g_free(entry);

    }



    g_free(qlist);

}
