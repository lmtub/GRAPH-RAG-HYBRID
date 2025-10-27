static void test_visitor_out_list(TestOutputVisitorData *data,

                                  const void *unused)

{

    const char *value_str = "list value";

    TestStructList *p, *head = NULL;

    const int max_items = 10;

    bool value_bool = true;

    int value_int = 10;

    Error *err = NULL;

    QListEntry *entry;

    QObject *obj;

    QList *qlist;

    int i;



    /* Build the list in reverse order... */

    for (i = 0; i < max_items; i++) {

        p = g_malloc0(sizeof(*p));

        p->value = g_malloc0(sizeof(*p->value));

        p->value->integer = value_int + (max_items - i - 1);

        p->value->boolean = value_bool;

        p->value->string = g_strdup(value_str);



        p->next = head;

        head = p;

    }



    visit_type_TestStructList(data->ov, &head, NULL, &err);

    g_assert(!err);



    obj = qmp_output_get_qobject(data->qov);

    g_assert(obj != NULL);

    g_assert(qobject_type(obj) == QTYPE_QLIST);



    qlist = qobject_to_qlist(obj);

    g_assert(!qlist_empty(qlist));



    /* ...and ensure that the visitor sees it in order */

    i = 0;

    QLIST_FOREACH_ENTRY(qlist, entry) {

        QDict *qdict;



        g_assert(qobject_type(entry->value) == QTYPE_QDICT);

        qdict = qobject_to_qdict(entry->value);

        g_assert_cmpint(qdict_size(qdict), ==, 3);

        g_assert_cmpint(qdict_get_int(qdict, "integer"), ==, value_int + i);

        g_assert_cmpint(qdict_get_bool(qdict, "boolean"), ==, value_bool);

        g_assert_cmpstr(qdict_get_str(qdict, "string"), ==, value_str);

        i++;

    }

    g_assert_cmpint(i, ==, max_items);



    QDECREF(qlist);

    qapi_free_TestStructList(head);

}
