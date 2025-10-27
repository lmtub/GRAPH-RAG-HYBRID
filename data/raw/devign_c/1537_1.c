static void event_test_emit(test_QAPIEvent event, QDict *d, Error **errp)

{

    QObject *obj;

    QDict *t;

    int64_t s, ms;



    /* Verify that we have timestamp, then remove it to compare other fields */

    obj = qdict_get(d, "timestamp");

    g_assert(obj);

    t = qobject_to_qdict(obj);

    g_assert(t);

    obj = qdict_get(t, "seconds");

    g_assert(obj && qobject_type(obj) == QTYPE_QINT);

    s = qint_get_int(qobject_to_qint(obj));

    obj = qdict_get(t, "microseconds");

    g_assert(obj && qobject_type(obj) == QTYPE_QINT);

    ms = qint_get_int(qobject_to_qint(obj));

    if (s == -1) {

        g_assert(ms == -1);

    } else {

        g_assert(ms >= 0 && ms <= 999999);

    }

    g_assert(qdict_size(t) == 2);



    qdict_del(d, "timestamp");



    g_assert(qdict_cmp_simple(d, test_event_data->expect));



}
