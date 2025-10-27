static void test_after_failed_device_add(void)

{

    QDict *response;

    QDict *error;



    qtest_start("-drive if=none,id=drive0");



    /* Make device_add fail.  If this leaks the virtio-blk-pci device then a

     * reference to drive0 will also be held (via qdev properties).

     */

    response = qmp("{'execute': 'device_add',"

                   " 'arguments': {"

                   "   'driver': 'virtio-blk-pci',"

                   "   'drive': 'drive0'"

                   "}}");

    g_assert(response);

    error = qdict_get_qdict(response, "error");

    g_assert_cmpstr(qdict_get_try_str(error, "class"), ==, "GenericError");

    QDECREF(response);



    /* Delete the drive */

    drive_del();



    /* Try to re-add the drive.  This fails with duplicate IDs if a leaked

     * virtio-blk-pci exists that holds a reference to the old drive0.

     */

    drive_add();



    qtest_end();

}
