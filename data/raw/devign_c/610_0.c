static void qdict_destroy_obj(QObject *obj)

{

    int i;

    QDict *qdict;



    assert(obj != NULL);

    qdict = qobject_to_qdict(obj);



    for (i = 0; i < QDICT_BUCKET_MAX; i++) {

        QDictEntry *entry = QLIST_FIRST(&qdict->table[i]);

        while (entry) {

            QDictEntry *tmp = QLIST_NEXT(entry, next);

            QLIST_REMOVE(entry, next);

            qentry_destroy(entry);

            entry = tmp;

        }

    }



    g_free(qdict);

}
