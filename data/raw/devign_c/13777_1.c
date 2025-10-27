static void test_machine(const void *data)

{

    const testdef_t *test = data;

    char *args;

    char tmpname[] = "/tmp/qtest-boot-serial-XXXXXX";

    int fd;



    fd = mkstemp(tmpname);

    g_assert(fd != -1);



    args = g_strdup_printf("-M %s,accel=tcg -chardev file,id=serial0,path=%s"

                           " -serial chardev:serial0 %s", test->machine,

                           tmpname, test->extra);



    qtest_start(args);

    unlink(tmpname);



    check_guest_output(test, fd);

    qtest_quit(global_qtest);



    g_free(args);

    close(fd);

}
