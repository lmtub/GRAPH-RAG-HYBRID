static void test_event_c(TestEventData *data,

                         const void *unused)

{

    QDict *d, *d_data, *d_b;



    UserDefOne b;

    UserDefZero z;

    z.integer = 2;

    b.base = &z;

    b.string = g_strdup("test1");

    b.has_enum1 = false;



    d_b = qdict_new();

    qdict_put(d_b, "integer", qint_from_int(2));

    qdict_put(d_b, "string", qstring_from_str("test1"));



    d_data = qdict_new();

    qdict_put(d_data, "a", qint_from_int(1));

    qdict_put(d_data, "b", d_b);

    qdict_put(d_data, "c", qstring_from_str("test2"));



    d = data->expect;

    qdict_put(d, "event", qstring_from_str("EVENT_C"));

    qdict_put(d, "data", d_data);



    qapi_event_send_event_c(true, 1, true, &b, "test2", &error_abort);



    g_free(b.string);

}
