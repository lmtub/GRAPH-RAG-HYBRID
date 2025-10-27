static void pc_dimm_plug(HotplugHandler *hotplug_dev,
                         DeviceState *dev, Error **errp)
{
    int slot;
    HotplugHandlerClass *hhc;
    Error *local_err = NULL;
    PCMachineState *pcms = PC_MACHINE(hotplug_dev);
    MachineState *machine = MACHINE(hotplug_dev);
    PCDIMMDevice *dimm = PC_DIMM(dev);
    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(dimm);
    MemoryRegion *mr = ddc->get_memory_region(dimm);
    uint64_t addr = object_property_get_int(OBJECT(dimm), PC_DIMM_ADDR_PROP,
                                            &local_err);
    if (local_err) {
    addr = pc_dimm_get_free_addr(pcms->hotplug_memory_base,
                                 memory_region_size(&pcms->hotplug_memory),
                                 !addr ? NULL : &addr,
                                 memory_region_size(mr), &local_err);
    if (local_err) {
    object_property_set_int(OBJECT(dev), addr, PC_DIMM_ADDR_PROP, &local_err);
    if (local_err) {
    trace_mhp_pc_dimm_assigned_address(addr);
    slot = object_property_get_int(OBJECT(dev), PC_DIMM_SLOT_PROP, &local_err);
    if (local_err) {
    slot = pc_dimm_get_free_slot(slot == PC_DIMM_UNASSIGNED_SLOT ? NULL : &slot,
                                 machine->ram_slots, &local_err);
    if (local_err) {
    object_property_set_int(OBJECT(dev), slot, PC_DIMM_SLOT_PROP, &local_err);
    if (local_err) {
    trace_mhp_pc_dimm_assigned_slot(slot);
    if (!pcms->acpi_dev) {
        error_setg(&local_err,
                   "memory hotplug is not enabled: missing acpi device");
    memory_region_add_subregion(&pcms->hotplug_memory,
                                addr - pcms->hotplug_memory_base, mr);
    vmstate_register_ram(mr, dev);
    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);
    hhc->plug(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);
out:
    error_propagate(errp, local_err);