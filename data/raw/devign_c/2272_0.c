static void pc_dimm_post_plug(HotplugHandler *hotplug_dev,

                              DeviceState *dev, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);



    if (object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM)) {

        nvdimm_acpi_hotplug(&pcms->acpi_nvdimm_state);

    }

}
