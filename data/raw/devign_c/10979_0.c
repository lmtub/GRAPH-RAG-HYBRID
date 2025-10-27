START_TEST(simple_string)

{

    int i;

    struct {

        const char *encoded;

        const char *decoded;

    } test_cases[] = {

        { "\"hello world\"", "hello world" },

        { "\"the quick brown fox jumped over the fence\"",

          "the quick brown fox jumped over the fence" },

        {}

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_json(test_cases[i].encoded);



        fail_unless(obj != NULL);

        fail_unless(qobject_type(obj) == QTYPE_QSTRING);

        

        str = qobject_to_qstring(obj);

        fail_unless(strcmp(qstring_get_str(str), test_cases[i].decoded) == 0);



        str = qobject_to_json(obj);

        fail_unless(strcmp(qstring_get_str(str), test_cases[i].encoded) == 0);



        qobject_decref(obj);

        

        QDECREF(str);

    }

}
