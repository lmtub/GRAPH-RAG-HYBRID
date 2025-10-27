static void test_validate_qmp_introspect(TestInputVisitorData *data,

                                           const void *unused)

{

    do_test_validate_qmp_introspect(data, test_qmp_schema_json);

    do_test_validate_qmp_introspect(data, qmp_schema_json);

}
