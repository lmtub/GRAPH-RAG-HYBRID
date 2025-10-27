QList *qlist_new(void)

{

    QList *qlist;



    qlist = g_malloc(sizeof(*qlist));

    QTAILQ_INIT(&qlist->head);

    QOBJECT_INIT(qlist, &qlist_type);



    return qlist;

}
