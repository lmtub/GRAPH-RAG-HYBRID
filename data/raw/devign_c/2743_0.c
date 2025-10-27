static Suite *qfloat_suite(void)

{

    Suite *s;

    TCase *qfloat_public_tcase;



    s = suite_create("QFloat test-suite");



    qfloat_public_tcase = tcase_create("Public Interface");

    suite_add_tcase(s, qfloat_public_tcase);

    tcase_add_test(qfloat_public_tcase, qfloat_from_double_test);

    tcase_add_test(qfloat_public_tcase, qfloat_destroy_test);



    return s;

}
