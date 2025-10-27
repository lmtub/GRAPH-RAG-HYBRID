static void spapr_machine_device_plug(HotplugHandler *hotplug_dev,

                                      DeviceState *dev, Error **errp)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(qdev_get_machine());



    if (object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        uint32_t node;



        if (!smc->dr_lmb_enabled) {

            error_setg(errp, "Memory hotplug not supported for this machine");

            return;

        }

        node = object_property_get_int(OBJECT(dev), PC_DIMM_NODE_PROP, errp);

        if (*errp) {

            return;

        }



        spapr_memory_plug(hotplug_dev, dev, node, errp);

    }

}
