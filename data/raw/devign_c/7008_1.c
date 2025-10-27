START_TEST(escaped_string)

{

    int i;

    struct {

        const char *encoded;

        const char *decoded;

        int skip;

    } test_cases[] = {

        { "\"\\\"\"", "\"" },

        { "\"hello world \\\"embedded string\\\"\"",

          "hello world \"embedded string\"" },

        { "\"hello world\\nwith new line\"", "hello world\nwith new line" },

        { "\"single byte utf-8 \\u0020\"", "single byte utf-8  ", .skip = 1 },

        { "\"double byte utf-8 \\u00A2\"", "double byte utf-8 \xc2\xa2" },

        { "\"triple byte utf-8 \\u20AC\"", "triple byte utf-8 \xe2\x82\xac" },

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



        if (test_cases[i].skip == 0) {

            str = qobject_to_json(obj);

            fail_unless(strcmp(qstring_get_str(str), test_cases[i].encoded) == 0);



            qobject_decref(obj);

        }



        QDECREF(str);

    }

}
