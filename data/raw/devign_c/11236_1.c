static void test_visitor_in_null(TestInputVisitorData *data,

                                 const void *unused)

{

    Visitor *v;

    Error *err = NULL;

    char *tmp;



    /*

     * FIXME: Since QAPI doesn't know the 'null' type yet, we can't

     * test visit_type_null() by reading into a QAPI struct then

     * checking that it was populated correctly.  The best we can do

     * for now is ensure that we consumed null from the input, proven

     * by the fact that we can't re-read the key; and that we detect

     * when input is not null.

     */



    v = visitor_input_test_init(data, "{ 'a': null, 'b': '', 'c': null }");

    visit_start_struct(v, NULL, NULL, 0, &error_abort);

    visit_type_null(v, "a", &error_abort);

    visit_type_null(v, "b", &err);

    error_free_or_abort(&err);

    visit_type_str(v, "c", &tmp, &err);

    g_assert(!tmp);

    error_free_or_abort(&err);

    visit_check_struct(v, &error_abort);

    visit_end_struct(v, NULL);

}
