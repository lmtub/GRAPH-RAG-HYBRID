static Suite *qjson_suite(void)

{

    Suite *suite;

    TCase *string_literals, *number_literals, *keyword_literals;

    TCase *dicts, *lists, *whitespace, *varargs;



    string_literals = tcase_create("String Literals");

    tcase_add_test(string_literals, simple_string);

    tcase_add_test(string_literals, escaped_string);

    tcase_add_test(string_literals, single_quote_string);

    tcase_add_test(string_literals, vararg_string);



    number_literals = tcase_create("Number Literals");

    tcase_add_test(number_literals, simple_number);

    tcase_add_test(number_literals, float_number);

    tcase_add_test(number_literals, vararg_number);



    keyword_literals = tcase_create("Keywords");

    tcase_add_test(keyword_literals, keyword_literal);

    dicts = tcase_create("Objects");

    tcase_add_test(dicts, simple_dict);

    lists = tcase_create("Lists");

    tcase_add_test(lists, simple_list);



    whitespace = tcase_create("Whitespace");

    tcase_add_test(whitespace, simple_whitespace);



    varargs = tcase_create("Varargs");

    tcase_add_test(varargs, simple_varargs);



    suite = suite_create("QJSON test-suite");

    suite_add_tcase(suite, string_literals);

    suite_add_tcase(suite, number_literals);

    suite_add_tcase(suite, keyword_literals);

    suite_add_tcase(suite, dicts);

    suite_add_tcase(suite, lists);

    suite_add_tcase(suite, whitespace);

    suite_add_tcase(suite, varargs);



    return suite;

}
