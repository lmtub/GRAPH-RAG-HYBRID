static Suite *QList_suite(void)

{

    Suite *s;

    TCase *qlist_public_tcase;



    s = suite_create("QList suite");



    qlist_public_tcase = tcase_create("Public Interface");

    suite_add_tcase(s, qlist_public_tcase);

    tcase_add_test(qlist_public_tcase, qlist_new_test);

    tcase_add_test(qlist_public_tcase, qlist_append_test);

    tcase_add_test(qlist_public_tcase, qobject_to_qlist_test);

    tcase_add_test(qlist_public_tcase, qlist_destroy_test);

    tcase_add_test(qlist_public_tcase, qlist_iter_test);



    return s;

}
