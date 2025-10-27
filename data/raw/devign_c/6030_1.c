static void pc_cpu_unplug_request_cb(HotplugHandler *hotplug_dev,
                                     DeviceState *dev, Error **errp)
{
    int idx = -1;
    HotplugHandlerClass *hhc;
    Error *local_err = NULL;
    X86CPU *cpu = X86_CPU(dev);
    PCMachineState *pcms = PC_MACHINE(hotplug_dev);
    pc_find_cpu_slot(MACHINE(pcms), cpu->apic_id, &idx);
    assert(idx != -1);
    if (idx == 0) {
        error_setg(&local_err, "Boot CPU is unpluggable");
    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);
    hhc->unplug_request(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);
    if (local_err) {
 out:
    error_propagate(errp, local_err);