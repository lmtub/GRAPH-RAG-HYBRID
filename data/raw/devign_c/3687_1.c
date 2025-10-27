static void test_validate_fail_union_flat_no_discrim(TestInputVisitorData *data,

                                                     const void *unused)

{

    UserDefFlatUnion2 *tmp = NULL;

    Error *err = NULL;

    Visitor *v;



    /* test situation where discriminator field ('enum1' here) is missing */

    v = validate_test_init(data, "{ 'integer': 42, 'string': 'c', 'string1': 'd', 'string2': 'e' }");



    visit_type_UserDefFlatUnion2(v, &tmp, NULL, &err);

    g_assert(err);

    error_free(err);

    qapi_free_UserDefFlatUnion2(tmp);

}
