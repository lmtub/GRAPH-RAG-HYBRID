void do_info_mice(Monitor *mon, QObject **ret_data)

{

    QEMUPutMouseEntry *cursor;

    QList *mice_list;

    int current;



    mice_list = qlist_new();



    if (QTAILQ_EMPTY(&mouse_handlers)) {

        goto out;

    }



    current = QTAILQ_FIRST(&mouse_handlers)->index;



    QTAILQ_FOREACH(cursor, &mouse_handlers, node) {

        QObject *obj;

        obj = qobject_from_jsonf("{ 'name': %s, 'index': %d, 'current': %i }",

                                 cursor->qemu_put_mouse_event_name,

                                 cursor->index,

                                 cursor->index == current);

        qlist_append_obj(mice_list, obj);

    }



out:

    *ret_data = QOBJECT(mice_list);

}
