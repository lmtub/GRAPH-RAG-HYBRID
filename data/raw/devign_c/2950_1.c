static void machine_initfn(Object *obj)

{

    MachineState *ms = MACHINE(obj);



    ms->kernel_irqchip_allowed = true;

    ms->kvm_shadow_mem = -1;

    ms->dump_guest_core = true;




    object_property_add_str(obj, "accel",

                            machine_get_accel, machine_set_accel, NULL);

    object_property_set_description(obj, "accel",

                                    "Accelerator list",

                                    NULL);

    object_property_add_bool(obj, "kernel-irqchip",

                             NULL,

                             machine_set_kernel_irqchip,

                             NULL);

    object_property_set_description(obj, "kernel-irqchip",

                                    "Use KVM in-kernel irqchip",

                                    NULL);

    object_property_add(obj, "kvm-shadow-mem", "int",

                        machine_get_kvm_shadow_mem,

                        machine_set_kvm_shadow_mem,

                        NULL, NULL, NULL);

    object_property_set_description(obj, "kvm-shadow-mem",

                                    "KVM shadow MMU size",

                                    NULL);

    object_property_add_str(obj, "kernel",

                            machine_get_kernel, machine_set_kernel, NULL);

    object_property_set_description(obj, "kernel",

                                    "Linux kernel image file",

                                    NULL);

    object_property_add_str(obj, "initrd",

                            machine_get_initrd, machine_set_initrd, NULL);

    object_property_set_description(obj, "initrd",

                                    "Linux initial ramdisk file",

                                    NULL);

    object_property_add_str(obj, "append",

                            machine_get_append, machine_set_append, NULL);

    object_property_set_description(obj, "append",

                                    "Linux kernel command line",

                                    NULL);

    object_property_add_str(obj, "dtb",

                            machine_get_dtb, machine_set_dtb, NULL);

    object_property_set_description(obj, "dtb",

                                    "Linux kernel device tree file",

                                    NULL);

    object_property_add_str(obj, "dumpdtb",

                            machine_get_dumpdtb, machine_set_dumpdtb, NULL);

    object_property_set_description(obj, "dumpdtb",

                                    "Dump current dtb to a file and quit",

                                    NULL);

    object_property_add(obj, "phandle-start", "int",

                        machine_get_phandle_start,

                        machine_set_phandle_start,

                        NULL, NULL, NULL);

    object_property_set_description(obj, "phandle-start",

                                    "The first phandle ID we may generate dynamically",

                                    NULL);

    object_property_add_str(obj, "dt-compatible",

                            machine_get_dt_compatible,

                            machine_set_dt_compatible,

                            NULL);

    object_property_set_description(obj, "dt-compatible",

                                    "Overrides the \"compatible\" property of the dt root node",

                                    NULL);

    object_property_add_bool(obj, "dump-guest-core",

                             machine_get_dump_guest_core,

                             machine_set_dump_guest_core,

                             NULL);

    object_property_set_description(obj, "dump-guest-core",

                                    "Include guest memory in  a core dump",

                                    NULL);

    object_property_add_bool(obj, "mem-merge",

                             machine_get_mem_merge,

                             machine_set_mem_merge, NULL);

    object_property_set_description(obj, "mem-merge",

                                    "Enable/disable memory merge support",

                                    NULL);

    object_property_add_bool(obj, "usb",

                             machine_get_usb,

                             machine_set_usb, NULL);

    object_property_set_description(obj, "usb",

                                    "Set on/off to enable/disable usb",

                                    NULL);

    object_property_add_str(obj, "firmware",

                            machine_get_firmware,

                            machine_set_firmware, NULL);

    object_property_set_description(obj, "firmware",

                                    "Firmware image",

                                    NULL);

    object_property_add_bool(obj, "iommu",

                             machine_get_iommu,

                             machine_set_iommu, NULL);

    object_property_set_description(obj, "iommu",

                                    "Set on/off to enable/disable Intel IOMMU (VT-d)",

                                    NULL);



    /* Register notifier when init is done for sysbus sanity checks */

    ms->sysbus_notifier.notify = machine_init_notify;

    qemu_add_machine_init_done_notifier(&ms->sysbus_notifier);

}