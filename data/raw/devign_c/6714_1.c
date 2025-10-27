static int do_pcie_aer_inject_error(Monitor *mon,

                                    const QDict *qdict, QObject **ret_data)

{

    const char *id = qdict_get_str(qdict, "id");

    const char *error_name;

    uint32_t error_status;

    bool correctable;

    PCIDevice *dev;

    PCIEAERErr err;

    int ret;



    ret = pci_qdev_find_device(id, &dev);

    if (ret < 0) {

        monitor_printf(mon,

                       "id or pci device path is invalid or device not "

                       "found. %s\n", id);

        return ret;

    }

    if (!pci_is_express(dev)) {

        monitor_printf(mon, "the device doesn't support pci express. %s\n",

                       id);

        return -ENOSYS;

    }



    error_name = qdict_get_str(qdict, "error_status");

    if (pcie_aer_parse_error_string(error_name, &error_status, &correctable)) {

        char *e = NULL;

        error_status = strtoul(error_name, &e, 0);

        correctable = qdict_get_try_bool(qdict, "correctable", false);

        if (!e || *e != '\0') {

            monitor_printf(mon, "invalid error status value. \"%s\"",

                           error_name);

            return -EINVAL;

        }

    }

    err.status = error_status;

    err.source_id = pci_requester_id(dev);



    err.flags = 0;

    if (correctable) {

        err.flags |= PCIE_AER_ERR_IS_CORRECTABLE;

    }

    if (qdict_get_try_bool(qdict, "advisory_non_fatal", false)) {

        err.flags |= PCIE_AER_ERR_MAYBE_ADVISORY;

    }

    if (qdict_haskey(qdict, "header0")) {

        err.flags |= PCIE_AER_ERR_HEADER_VALID;

    }

    if (qdict_haskey(qdict, "prefix0")) {

        err.flags |= PCIE_AER_ERR_TLP_PREFIX_PRESENT;

    }



    err.header[0] = qdict_get_try_int(qdict, "header0", 0);

    err.header[1] = qdict_get_try_int(qdict, "header1", 0);

    err.header[2] = qdict_get_try_int(qdict, "header2", 0);

    err.header[3] = qdict_get_try_int(qdict, "header3", 0);



    err.prefix[0] = qdict_get_try_int(qdict, "prefix0", 0);

    err.prefix[1] = qdict_get_try_int(qdict, "prefix1", 0);

    err.prefix[2] = qdict_get_try_int(qdict, "prefix2", 0);

    err.prefix[3] = qdict_get_try_int(qdict, "prefix3", 0);



    ret = pcie_aer_inject_error(dev, &err);

    *ret_data = qobject_from_jsonf("{'id': %s, "

                                   "'root_bus': %s, 'bus': %d, 'devfn': %d, "

                                   "'ret': %d}",

                                   id, pci_root_bus_path(dev),

                                   pci_bus_num(dev->bus), dev->devfn,

                                   ret);

    assert(*ret_data);



    return 0;

}
