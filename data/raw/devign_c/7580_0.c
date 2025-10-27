static HotplugHandler *spapr_get_hotpug_handler(MachineState *machine,

                                             DeviceState *dev)

{

    if (object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM) ||

        object_dynamic_cast(OBJECT(dev), TYPE_SPAPR_CPU_CORE)) {

        return HOTPLUG_HANDLER(machine);

    }

    return NULL;

}
