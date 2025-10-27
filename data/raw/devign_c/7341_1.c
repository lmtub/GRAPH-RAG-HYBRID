static void virtio_scsi_bad_req(void)

{

    error_report("wrong size for virtio-scsi headers");

    exit(1);

}
