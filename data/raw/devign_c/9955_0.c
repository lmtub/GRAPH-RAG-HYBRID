static void test_event_d(TestEventData *data,

                         const void *unused)

{

    UserDefOne struct1;

    EventStructOne a;

    UserDefZero z;

    QDict *d, *d_data, *d_a, *d_struct1;



    z.integer = 2;

    struct1.base = &z;

    struct1.string = g_strdup("test1");

    struct1.has_enum1 = true;

    struct1.enum1 = ENUM_ONE_VALUE1;



    a.struct1 = &struct1;

    a.string = g_strdup("test2");

    a.has_enum2 = true;

    a.enum2 = ENUM_ONE_VALUE2;



    d_struct1 = qdict_new();

    qdict_put(d_struct1, "integer", qint_from_int(2));

    qdict_put(d_struct1, "string", qstring_from_str("test1"));

    qdict_put(d_struct1, "enum1", qstring_from_str("value1"));



    d_a = qdict_new();

    qdict_put(d_a, "struct1", d_struct1);

    qdict_put(d_a, "string", qstring_from_str("test2"));

    qdict_put(d_a, "enum2", qstring_from_str("value2"));



    d_data = qdict_new();

    qdict_put(d_data, "a", d_a);

    qdict_put(d_data, "b", qstring_from_str("test3"));

    qdict_put(d_data, "enum3", qstring_from_str("value3"));



    d = data->expect;

    qdict_put(d, "event", qstring_from_str("EVENT_D"));

    qdict_put(d, "data", d_data);



    qapi_event_send_event_d(&a, "test3", false, NULL, true, ENUM_ONE_VALUE3,

                           &error_abort);



    g_free(struct1.string);

    g_free(a.string);

}
