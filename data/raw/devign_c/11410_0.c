static void qvirtio_9p_stop(void)

{

    qtest_end();

    rmdir(test_share);

    g_free(test_share);

}
