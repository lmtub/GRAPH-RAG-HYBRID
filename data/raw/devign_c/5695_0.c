static void test_drive_del_device_del(void)

{

    /* Start with a drive used by a device that unplugs instantaneously */

    qtest_start("-drive if=none,id=drive0,file=null-co://,format=raw"

                " -device virtio-scsi-pci"

                " -device scsi-hd,drive=drive0,id=dev0");



    /*

     * Delete the drive, and then the device

     * Doing it in this order takes notoriously tricky special paths

     */

    drive_del();

    device_del();



    qtest_end();

}
