void kvm_arm_register_device(MemoryRegion *mr, uint64_t devid)

{

    KVMDevice *kd;



    if (!kvm_irqchip_in_kernel()) {

        return;

    }



    if (QSLIST_EMPTY(&kvm_devices_head)) {

        memory_listener_register(&devlistener, NULL);

        qemu_add_machine_init_done_notifier(&notify);

    }

    kd = g_new0(KVMDevice, 1);

    kd->mr = mr;

    kd->kda.id = devid;

    kd->kda.addr = -1;

    QSLIST_INSERT_HEAD(&kvm_devices_head, kd, entries);

    memory_region_ref(kd->mr);

}
