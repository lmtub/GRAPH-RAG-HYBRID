static QOSState *pci_test_start(int socket)

{

    const char *cmd = "-netdev socket,fd=%d,id=hs0 -device "

                      "virtio-net-pci,netdev=hs0";



    return qtest_pc_boot(cmd, socket);

}
