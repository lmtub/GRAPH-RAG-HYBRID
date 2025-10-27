static void escaped_string(void)

{

    int i;

    struct {

        const char *encoded;

        const char *decoded;

        int skip;

    } test_cases[] = {

        { "\"\\b\"", "\b" },

        { "\"\\f\"", "\f" },

        { "\"\\n\"", "\n" },

        { "\"\\r\"", "\r" },

        { "\"\\t\"", "\t" },

        { "\"/\"", "/" },

        { "\"\\/\"", "/", .skip = 1 },

        { "\"\\\\\"", "\\" },

        { "\"\\\"\"", "\"" },

        { "\"hello world \\\"embedded string\\\"\"",

          "hello world \"embedded string\"" },

        { "\"hello world\\nwith new line\"", "hello world\nwith new line" },

        { "\"single byte utf-8 \\u0020\"", "single byte utf-8  ", .skip = 1 },

        { "\"double byte utf-8 \\u00A2\"", "double byte utf-8 \xc2\xa2" },

        { "\"triple byte utf-8 \\u20AC\"", "triple byte utf-8 \xe2\x82\xac" },

        { "'\\b'", "\b", .skip = 1 },

        { "'\\f'", "\f", .skip = 1 },

        { "'\\n'", "\n", .skip = 1 },

        { "'\\r'", "\r", .skip = 1 },

        { "'\\t'", "\t", .skip = 1 },

        { "'\\/'", "/", .skip = 1 },

        { "'\\\\'", "\\", .skip = 1 },

        {}

    };



    for (i = 0; test_cases[i].encoded; i++) {

        QObject *obj;

        QString *str;



        obj = qobject_from_json(test_cases[i].encoded, NULL);

        str = qobject_to_qstring(obj);

        g_assert(str);

        g_assert_cmpstr(qstring_get_str(str), ==, test_cases[i].decoded);



        if (test_cases[i].skip == 0) {

            str = qobject_to_json(obj);

            g_assert_cmpstr(qstring_get_str(str), ==, test_cases[i].encoded);

            qobject_decref(obj);

        }



        QDECREF(str);

    }

}
