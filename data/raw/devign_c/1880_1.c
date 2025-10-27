static void pci_bridge_update_mappings(PCIBridge *br)

{

    /* Make updates atomic to: handle the case of one VCPU updating the bridge

     * while another accesses an unaffected region. */

    memory_region_transaction_begin();

    pci_bridge_region_cleanup(br);

    pci_bridge_region_init(br);

    memory_region_transaction_commit();

}
