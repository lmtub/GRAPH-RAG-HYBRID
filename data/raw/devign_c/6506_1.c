static void test_visitor_in_errors(TestInputVisitorData *data,

                                   const void *unused)

{

    TestStruct *p = NULL;

    Error *err = NULL;

    Visitor *v;



    v = visitor_input_test_init(data, "{ 'integer': false, 'boolean': 'foo', 'string': -42 }");



    visit_type_TestStruct(v, &p, NULL, &err);

    error_free_or_abort(&err);

    /* FIXME - a failed parse should not leave a partially-allocated p

     * for us to clean up; this could cause callers to leak memory. */

    g_assert(p->string == NULL);



    g_free(p->string);

    g_free(p);

}
