QObject *qlist_peek(QList *qlist)

{

    QListEntry *entry;

    QObject *ret;



    if (qlist == NULL || QTAILQ_EMPTY(&qlist->head)) {

        return NULL;

    }



    entry = QTAILQ_FIRST(&qlist->head);



    ret = entry->value;



    return ret;

}
