static Suite *qint_suite(void)

{

    Suite *s;

    TCase *qint_public_tcase;



    s = suite_create("QInt test-suite");



    qint_public_tcase = tcase_create("Public Interface");

    suite_add_tcase(s, qint_public_tcase);

    tcase_add_test(qint_public_tcase, qint_from_int_test);

    tcase_add_test(qint_public_tcase, qint_destroy_test);

    tcase_add_test(qint_public_tcase, qint_from_int64_test);

    tcase_add_test(qint_public_tcase, qint_get_int_test);

    tcase_add_test(qint_public_tcase, qobject_to_qint_test);



    return s;

}
