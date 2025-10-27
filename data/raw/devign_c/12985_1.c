void hmp_pcie_aer_inject_error(Monitor *mon, const QDict *qdict)

{

    QObject *data;

    int devfn;



    if (do_pcie_aer_inject_error(mon, qdict, &data) < 0) {

        return;

    }



    qdict = qobject_to_qdict(data);

    assert(qdict);



    devfn = (int)qdict_get_int(qdict, "devfn");

    monitor_printf(mon, "OK id: %s root bus: %s, bus: %x devfn: %x.%x\n",

                   qdict_get_str(qdict, "id"),

                   qdict_get_str(qdict, "root_bus"),

                   (int) qdict_get_int(qdict, "bus"),

                   PCI_SLOT(devfn), PCI_FUNC(devfn));

}
