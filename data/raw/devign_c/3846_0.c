static void test_unaligned_write_same(void)

{

    QVirtIOSCSI *vs;

    uint8_t buf[512] = { 0 };

    const uint8_t write_same_cdb[CDB_SIZE] = { 0x41, 0x00, 0x00, 0x00, 0x00,

                                               0x01, 0x00, 0x00, 0x02, 0x00 };



    qvirtio_scsi_start("-drive file=blkdebug::null-co://,if=none,id=dr1"

                       ",format=raw,file.align=4k "

                       "-device scsi-disk,drive=dr1,lun=0,scsi-id=1");

    vs = qvirtio_scsi_pci_init(PCI_SLOT);



    g_assert_cmphex(0, ==,

        virtio_scsi_do_command(vs, write_same_cdb, NULL, 0, buf, 512));



    qvirtio_scsi_pci_free(vs);

    qvirtio_scsi_stop();

}
