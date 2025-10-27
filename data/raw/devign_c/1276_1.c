static void test_dispatch_cmd_io(void)

{

    QDict *req = qdict_new();

    QDict *args = qdict_new();

    QDict *args3 = qdict_new();

    QDict *ud1a = qdict_new();

    QDict *ud1b = qdict_new();

    QDict *ret, *ret_dict, *ret_dict_dict, *ret_dict_dict_userdef;

    QDict *ret_dict_dict2, *ret_dict_dict2_userdef;

    QInt *ret3;



    qdict_put_obj(ud1a, "integer", QOBJECT(qint_from_int(42)));

    qdict_put_obj(ud1a, "string", QOBJECT(qstring_from_str("hello")));

    qdict_put_obj(ud1b, "integer", QOBJECT(qint_from_int(422)));

    qdict_put_obj(ud1b, "string", QOBJECT(qstring_from_str("hello2")));

    qdict_put_obj(args, "ud1a", QOBJECT(ud1a));

    qdict_put_obj(args, "ud1b", QOBJECT(ud1b));

    qdict_put_obj(req, "arguments", QOBJECT(args));

    qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd2")));



    ret = qobject_to_qdict(test_qmp_dispatch(req));



    assert(!strcmp(qdict_get_str(ret, "string"), "blah1"));

    ret_dict = qdict_get_qdict(ret, "dict");

    assert(!strcmp(qdict_get_str(ret_dict, "string"), "blah2"));

    ret_dict_dict = qdict_get_qdict(ret_dict, "dict");

    ret_dict_dict_userdef = qdict_get_qdict(ret_dict_dict, "userdef");

    assert(qdict_get_int(ret_dict_dict_userdef, "integer") == 42);

    assert(!strcmp(qdict_get_str(ret_dict_dict_userdef, "string"), "hello"));

    assert(!strcmp(qdict_get_str(ret_dict_dict, "string"), "blah3"));

    ret_dict_dict2 = qdict_get_qdict(ret_dict, "dict2");

    ret_dict_dict2_userdef = qdict_get_qdict(ret_dict_dict2, "userdef");

    assert(qdict_get_int(ret_dict_dict2_userdef, "integer") == 422);

    assert(!strcmp(qdict_get_str(ret_dict_dict2_userdef, "string"), "hello2"));

    assert(!strcmp(qdict_get_str(ret_dict_dict2, "string"), "blah4"));

    QDECREF(ret);



    qdict_put(args3, "a", qint_from_int(66));

    qdict_put(req, "arguments", args3);

    qdict_put(req, "execute", qstring_from_str("user_def_cmd3"));



    ret3 = qobject_to_qint(test_qmp_dispatch(req));

    assert(qint_get_int(ret3) == 66);

    QDECREF(ret);



    QDECREF(req);

}
