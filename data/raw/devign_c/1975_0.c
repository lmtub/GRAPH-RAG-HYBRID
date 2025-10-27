int main(int argc, char **argv)

{

    TestInputVisitorData testdata;



    g_test_init(&argc, &argv, NULL);



    validate_test_add("/visitor/input-strict/pass/struct",

                       &testdata, test_validate_struct);

    validate_test_add("/visitor/input-strict/pass/struct-nested",

                       &testdata, test_validate_struct_nested);

    validate_test_add("/visitor/input-strict/pass/list",

                       &testdata, test_validate_list);

    validate_test_add("/visitor/input-strict/pass/union",

                       &testdata, test_validate_union);

    validate_test_add("/visitor/input-strict/pass/union-flat",

                       &testdata, test_validate_union_flat);

    validate_test_add("/visitor/input-strict/pass/union-anon",

                       &testdata, test_validate_union_anon);

    validate_test_add("/visitor/input-strict/fail/struct",

                       &testdata, test_validate_fail_struct);

    validate_test_add("/visitor/input-strict/fail/struct-nested",

                       &testdata, test_validate_fail_struct_nested);

    validate_test_add("/visitor/input-strict/fail/list",

                       &testdata, test_validate_fail_list);

    validate_test_add("/visitor/input-strict/fail/union",

                       &testdata, test_validate_fail_union);

    validate_test_add("/visitor/input-strict/fail/union-flat",

                       &testdata, test_validate_fail_union_flat);

    validate_test_add("/visitor/input-strict/fail/union-flat-no-discriminator",

                       &testdata, test_validate_fail_union_flat_no_discrim);

    validate_test_add("/visitor/input-strict/fail/union-anon",

                       &testdata, test_validate_fail_union_anon);



    g_test_run();



    return 0;

}
