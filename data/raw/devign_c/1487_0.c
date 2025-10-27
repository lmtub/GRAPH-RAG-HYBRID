int main(void)

{

    int nf;

    Suite *s;

    SRunner *sr;



    s = qfloat_suite();

    sr = srunner_create(s);



    srunner_run_all(sr, CK_NORMAL);

    nf = srunner_ntests_failed(sr);

    srunner_free(sr);



    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;

}
