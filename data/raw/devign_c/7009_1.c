static void pc_dimm_unplug_request(HotplugHandler *hotplug_dev,

                                   DeviceState *dev, Error **errp)

{

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);



    if (!pcms->acpi_dev) {

        error_setg(&local_err,

                   "memory hotplug is not enabled: missing acpi device");

        goto out;

    }



    if (object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM)) {

        error_setg(&local_err,

                   "nvdimm device hot unplug is not supported yet.");

        goto out;

    }



    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

    hhc->unplug_request(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);



out:

    error_propagate(errp, local_err);

}
