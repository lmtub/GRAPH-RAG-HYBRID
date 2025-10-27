static void test_migrate(void)

{

    char *uri = g_strdup_printf("unix:%s/migsocket", tmpfs);

    QTestState *global = global_qtest, *from, *to;

    unsigned char dest_byte_a, dest_byte_b, dest_byte_c, dest_byte_d;

    gchar *cmd;

    QDict *rsp;



    char *bootpath = g_strdup_printf("%s/bootsect", tmpfs);

    FILE *bootfile = fopen(bootpath, "wb");



    got_stop = false;

    g_assert_cmpint(fwrite(bootsect, 512, 1, bootfile), ==, 1);

    fclose(bootfile);



    cmd = g_strdup_printf("-machine accel=kvm:tcg -m 150M"

                          " -name pcsource,debug-threads=on"

                          " -serial file:%s/src_serial"

                          " -drive file=%s,format=raw",

                          tmpfs, bootpath);

    from = qtest_start(cmd);

    g_free(cmd);



    cmd = g_strdup_printf("-machine accel=kvm:tcg -m 150M"

                          " -name pcdest,debug-threads=on"

                          " -serial file:%s/dest_serial"

                          " -drive file=%s,format=raw"

                          " -incoming %s",

                          tmpfs, bootpath, uri);

    to = qtest_init(cmd);

    g_free(cmd);



    global_qtest = from;

    rsp = qmp("{ 'execute': 'migrate-set-capabilities',"

                  "'arguments': { "

                      "'capabilities': [ {"

                          "'capability': 'postcopy-ram',"

                          "'state': true } ] } }");

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);



    global_qtest = to;

    rsp = qmp("{ 'execute': 'migrate-set-capabilities',"

                  "'arguments': { "

                      "'capabilities': [ {"

                          "'capability': 'postcopy-ram',"

                          "'state': true } ] } }");

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);



    /* We want to pick a speed slow enough that the test completes

     * quickly, but that it doesn't complete precopy even on a slow

     * machine, so also set the downtime.

     */

    global_qtest = from;

    rsp = qmp("{ 'execute': 'migrate_set_speed',"

              "'arguments': { 'value': 100000000 } }");

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);



    /* 1ms downtime - it should never finish precopy */

    rsp = qmp("{ 'execute': 'migrate_set_downtime',"

              "'arguments': { 'value': 0.001 } }");

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);





    /* Wait for the first serial output from the source */

    wait_for_serial("src_serial");



    cmd = g_strdup_printf("{ 'execute': 'migrate',"

                          "'arguments': { 'uri': '%s' } }",

                          uri);

    rsp = qmp(cmd);

    g_free(cmd);

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);



    wait_for_migration_pass();



    rsp = return_or_event(qmp("{ 'execute': 'migrate-start-postcopy' }"));

    g_assert(qdict_haskey(rsp, "return"));

    QDECREF(rsp);



    if (!got_stop) {

        qmp_eventwait("STOP");

    }



    global_qtest = to;

    qmp_eventwait("RESUME");



    wait_for_serial("dest_serial");

    global_qtest = from;

    wait_for_migration_complete();



    qtest_quit(from);



    global_qtest = to;



    qtest_memread(to, start_address, &dest_byte_a, 1);



    /* Destination still running, wait for a byte to change */

    do {

        qtest_memread(to, start_address, &dest_byte_b, 1);

        usleep(10 * 1000);

    } while (dest_byte_a == dest_byte_b);



    qmp("{ 'execute' : 'stop'}");

    /* With it stopped, check nothing changes */

    qtest_memread(to, start_address, &dest_byte_c, 1);

    sleep(1);

    qtest_memread(to, start_address, &dest_byte_d, 1);

    g_assert_cmpint(dest_byte_c, ==, dest_byte_d);



    check_guests_ram();



    qtest_quit(to);

    g_free(uri);



    global_qtest = global;



    cleanup("bootsect");

    cleanup("migsocket");

    cleanup("src_serial");

    cleanup("dest_serial");

}
