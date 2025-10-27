void mkimg(const char *file, const char *fmt, unsigned size_mb)

{

    gchar *cli;

    bool ret;

    int rc;

    GError *err = NULL;

    char *qemu_img_path;

    gchar *out, *out2;

    char *abs_path;



    qemu_img_path = getenv("QTEST_QEMU_IMG");

    abs_path = realpath(qemu_img_path, NULL);

    assert(qemu_img_path);



    cli = g_strdup_printf("%s create -f %s %s %uM", abs_path,

                          fmt, file, size_mb);

    ret = g_spawn_command_line_sync(cli, &out, &out2, &rc, &err);

    if (err) {

        fprintf(stderr, "%s\n", err->message);

        g_error_free(err);

    }

    g_assert(ret && !err);



    /* In glib 2.34, we have g_spawn_check_exit_status. in 2.12, we don't.

     * glib 2.43.91 implementation assumes that any non-zero is an error for

     * windows, but uses extra precautions for Linux. However,

     * 0 is only possible if the program exited normally, so that should be

     * sufficient for our purposes on all platforms, here. */

    if (rc) {

        fprintf(stderr, "qemu-img returned status code %d\n", rc);

    }

    g_assert(!rc);



    g_free(out);

    g_free(out2);

    g_free(cli);

    free(abs_path);

}
