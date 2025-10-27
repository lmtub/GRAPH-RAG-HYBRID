static void hotplug(void)

{

    qtest_start("-device virtio-net-pci");



    qpci_plug_device_test("virtio-net-pci", "net1", PCI_SLOT_HP, NULL);

    qpci_unplug_acpi_device_test("net1", PCI_SLOT_HP);



    test_end();

}
