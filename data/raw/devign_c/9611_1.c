static bool acpi_get_mcfg(AcpiMcfgInfo *mcfg)

{

    Object *pci_host;

    QObject *o;



    pci_host = acpi_get_i386_pci_host();

    g_assert(pci_host);



    o = object_property_get_qobject(pci_host, PCIE_HOST_MCFG_BASE, NULL);

    if (!o) {

        return false;

    }

    mcfg->mcfg_base = qnum_get_int(qobject_to_qnum(o));

    qobject_decref(o);



    o = object_property_get_qobject(pci_host, PCIE_HOST_MCFG_SIZE, NULL);

    assert(o);

    mcfg->mcfg_size = qnum_get_int(qobject_to_qnum(o));

    qobject_decref(o);

    return true;

}
