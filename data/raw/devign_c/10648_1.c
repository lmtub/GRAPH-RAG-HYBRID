static void vmgenid_query_monitor_test(void)

{

    QemuUUID expected, measured;

    gchar *cmd;



    g_assert(qemu_uuid_parse(VGID_GUID, &expected) == 0);



    cmd = g_strdup_printf("-machine accel=tcg -device vmgenid,id=testvgid,"

                          "guid=%s", VGID_GUID);

    qtest_start(cmd);



    /* Read the GUID via the monitor */

    read_guid_from_monitor(&measured);

    g_assert(memcmp(measured.data, expected.data, sizeof(measured.data)) == 0);



    qtest_quit(global_qtest);

    g_free(cmd);

}
