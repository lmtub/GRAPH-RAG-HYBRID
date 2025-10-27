static void qdict_crumple_test_recursive(void)

{

    QDict *src, *dst, *rule, *vnc, *acl, *listen;

    QObject *child, *res;

    QList *rules;



    src = qdict_new();

    qdict_put(src, "vnc.listen.addr", qstring_from_str("127.0.0.1"));

    qdict_put(src, "vnc.listen.port", qstring_from_str("5901"));

    qdict_put(src, "vnc.acl.rules.0.match", qstring_from_str("fred"));

    qdict_put(src, "vnc.acl.rules.0.policy", qstring_from_str("allow"));

    qdict_put(src, "vnc.acl.rules.1.match", qstring_from_str("bob"));

    qdict_put(src, "vnc.acl.rules.1.policy", qstring_from_str("deny"));

    qdict_put(src, "vnc.acl.default", qstring_from_str("deny"));

    qdict_put(src, "vnc.acl..name", qstring_from_str("acl0"));

    qdict_put(src, "vnc.acl.rule..name", qstring_from_str("acl0"));



    res = qdict_crumple(src, &error_abort);



    g_assert_cmpint(qobject_type(res), ==, QTYPE_QDICT);



    dst = qobject_to_qdict(res);



    g_assert_cmpint(qdict_size(dst), ==, 1);



    child = qdict_get(dst, "vnc");

    g_assert_cmpint(qobject_type(child), ==, QTYPE_QDICT);

    vnc = qobject_to_qdict(child);



    child = qdict_get(vnc, "listen");

    g_assert_cmpint(qobject_type(child), ==, QTYPE_QDICT);

    listen = qobject_to_qdict(child);

    g_assert_cmpstr("127.0.0.1", ==, qdict_get_str(listen, "addr"));

    g_assert_cmpstr("5901", ==, qdict_get_str(listen, "port"));



    child = qdict_get(vnc, "acl");

    g_assert_cmpint(qobject_type(child), ==, QTYPE_QDICT);

    acl = qobject_to_qdict(child);



    child = qdict_get(acl, "rules");

    g_assert_cmpint(qobject_type(child), ==, QTYPE_QLIST);

    rules = qobject_to_qlist(child);

    g_assert_cmpint(qlist_size(rules), ==, 2);



    rule = qobject_to_qdict(qlist_pop(rules));

    g_assert_cmpint(qdict_size(rule), ==, 2);

    g_assert_cmpstr("fred", ==, qdict_get_str(rule, "match"));

    g_assert_cmpstr("allow", ==, qdict_get_str(rule, "policy"));

    QDECREF(rule);



    rule = qobject_to_qdict(qlist_pop(rules));

    g_assert_cmpint(qdict_size(rule), ==, 2);

    g_assert_cmpstr("bob", ==, qdict_get_str(rule, "match"));

    g_assert_cmpstr("deny", ==, qdict_get_str(rule, "policy"));

    QDECREF(rule);



    /* With recursive crumpling, we should see all names unescaped */

    g_assert_cmpstr("acl0", ==, qdict_get_str(vnc, "acl.name"));

    child = qdict_get(vnc, "acl");

    g_assert_cmpint(qobject_type(child), ==, QTYPE_QDICT);

    acl = qdict_get_qdict(vnc, "acl");

    g_assert_cmpstr("acl0", ==, qdict_get_str(acl, "rule.name"));



    QDECREF(src);

    QDECREF(dst);

}
