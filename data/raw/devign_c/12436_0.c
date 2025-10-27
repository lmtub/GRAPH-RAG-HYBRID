static void pci_nop(void)

{

    qvirtio_scsi_start(NULL);

    qvirtio_scsi_stop();

}
