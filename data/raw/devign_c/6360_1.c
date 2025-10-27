static void test_interface_impl(const char *type)

{

    Object *obj = object_new(type);

    TestIf *iobj = TEST_IF(obj);

    TestIfClass *ioc = TEST_IF_GET_CLASS(iobj);



    g_assert(iobj);

    g_assert(ioc->test == PATTERN);


}