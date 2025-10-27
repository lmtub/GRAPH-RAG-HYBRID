static QOSState *qvirtio_scsi_start(const char *extra_opts)

{

    const char *cmd = "-drive id=drv0,if=none,file=/dev/null,format=raw "

                      "-device virtio-scsi-pci,id=vs0 "

                      "-device scsi-hd,bus=vs0.0,drive=drv0 %s";



    return qtest_pc_boot(cmd, extra_opts ? : "");

}
