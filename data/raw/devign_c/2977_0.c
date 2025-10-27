static void test_visitor_out_struct_nested(TestOutputVisitorData *data,

                                           const void *unused)

{

    int64_t value = 42;

    Error *err = NULL;

    UserDefNested *ud2;

    QObject *obj;

    QDict *qdict, *dict1, *dict2, *dict3, *userdef;

    const char *string = "user def string";

    const char *strings[] = { "forty two", "forty three", "forty four",

                              "forty five" };



    ud2 = g_malloc0(sizeof(*ud2));

    ud2->string0 = g_strdup(strings[0]);



    ud2->dict1.string1 = g_strdup(strings[1]);

    ud2->dict1.dict2.userdef1 = g_malloc0(sizeof(UserDefOne));

    ud2->dict1.dict2.userdef1->string = g_strdup(string);

    ud2->dict1.dict2.userdef1->base = g_new0(UserDefZero, 1);

    ud2->dict1.dict2.userdef1->base->integer = value;

    ud2->dict1.dict2.string2 = g_strdup(strings[2]);



    ud2->dict1.has_dict3 = true;

    ud2->dict1.dict3.userdef2 = g_malloc0(sizeof(UserDefOne));

    ud2->dict1.dict3.userdef2->string = g_strdup(string);

    ud2->dict1.dict3.userdef2->base = g_new0(UserDefZero, 1);

    ud2->dict1.dict3.userdef2->base->integer = value;

    ud2->dict1.dict3.string3 = g_strdup(strings[3]);



    visit_type_UserDefNested(data->ov, &ud2, "unused", &err);

    g_assert(!err);



    obj = qmp_output_get_qobject(data->qov);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QDICT);



    qdict = qobject_to_qdict(obj);

    g_assert_cmpint(qdict_size(qdict), ==, 2);

    g_assert_cmpstr(qdict_get_str(qdict, "string0"), ==, strings[0]);



    dict1 = qdict_get_qdict(qdict, "dict1");

    g_assert_cmpint(qdict_size(dict1), ==, 3);

    g_assert_cmpstr(qdict_get_str(dict1, "string1"), ==, strings[1]);



    dict2 = qdict_get_qdict(dict1, "dict2");

    g_assert_cmpint(qdict_size(dict2), ==, 2);

    g_assert_cmpstr(qdict_get_str(dict2, "string2"), ==, strings[2]);

    userdef = qdict_get_qdict(dict2, "userdef1");

    g_assert_cmpint(qdict_size(userdef), ==, 2);

    g_assert_cmpint(qdict_get_int(userdef, "integer"), ==, value);

    g_assert_cmpstr(qdict_get_str(userdef, "string"), ==, string);



    dict3 = qdict_get_qdict(dict1, "dict3");

    g_assert_cmpint(qdict_size(dict3), ==, 2);

    g_assert_cmpstr(qdict_get_str(dict3, "string3"), ==, strings[3]);

    userdef = qdict_get_qdict(dict3, "userdef2");

    g_assert_cmpint(qdict_size(userdef), ==, 2);

    g_assert_cmpint(qdict_get_int(userdef, "integer"), ==, value);

    g_assert_cmpstr(qdict_get_str(userdef, "string"), ==, string);



    QDECREF(qdict);

    qapi_free_UserDefNested(ud2);

}
