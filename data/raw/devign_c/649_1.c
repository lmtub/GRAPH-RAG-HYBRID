static void limits_nesting(void)

{

    enum { max_nesting = 1024 }; /* see qobject/json-streamer.c */

    char buf[2 * (max_nesting + 1) + 1];

    QObject *obj;



    obj = qobject_from_json(make_nest(buf, max_nesting), NULL);

    g_assert(obj != NULL);

    qobject_decref(obj);



    obj = qobject_from_json(make_nest(buf, max_nesting + 1), NULL);

    g_assert(obj == NULL);

}
