void ich9_pm_device_plug_cb(HotplugHandler *hotplug_dev, DeviceState *dev,

                            Error **errp)

{

    ICH9LPCState *lpc = ICH9_LPC_DEVICE(hotplug_dev);



    if (lpc->pm.acpi_memory_hotplug.is_enabled &&

        object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        acpi_memory_plug_cb(hotplug_dev, &lpc->pm.acpi_memory_hotplug,

                            dev, errp);

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_CPU)) {

        if (lpc->pm.cpu_hotplug_legacy) {

            legacy_acpi_cpu_plug_cb(hotplug_dev, &lpc->pm.gpe_cpu, dev, errp);

        } else {

            acpi_cpu_plug_cb(hotplug_dev, &lpc->pm.cpuhp_state, dev, errp);

        }

    } else {

        error_setg(errp, "acpi: device plug request for not supported device"

                   " type: %s", object_get_typename(OBJECT(dev)));

    }

}
