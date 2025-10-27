static void test_ide_none(void)

{

    char *argv[256];



    setup_common(argv, ARRAY_SIZE(argv));

    qtest_start(g_strjoinv(" ", argv));

    test_cmos();

    qtest_end();

}
