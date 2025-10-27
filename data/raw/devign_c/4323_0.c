void pc_basic_device_init(ISABus *isa_bus, qemu_irq *gsi,

                          ISADevice **rtc_state,

                          ISADevice **floppy,

                          bool no_vmport)

{

    int i;

    DriveInfo *fd[MAX_FD];

    DeviceState *hpet = NULL;

    int pit_isa_irq = 0;

    qemu_irq pit_alt_irq = NULL;

    qemu_irq rtc_irq = NULL;

    qemu_irq *a20_line;

    ISADevice *i8042, *port92, *vmmouse, *pit;

    qemu_irq *cpu_exit_irq;



    register_ioport_write(0x80, 1, 1, ioport80_write, NULL);



    register_ioport_write(0xf0, 1, 1, ioportF0_write, NULL);



    /*

     * Check if an HPET shall be created.

     *

     * Without KVM_CAP_PIT_STATE2, we cannot switch off the in-kernel PIT

     * when the HPET wants to take over. Thus we have to disable the latter.

     */

    if (!no_hpet && (!kvm_irqchip_in_kernel() || kvm_has_pit_state2())) {

        hpet = sysbus_try_create_simple("hpet", HPET_BASE, NULL);



        if (hpet) {

            for (i = 0; i < GSI_NUM_PINS; i++) {

                sysbus_connect_irq(sysbus_from_qdev(hpet), i, gsi[i]);

            }

            pit_isa_irq = -1;

            pit_alt_irq = qdev_get_gpio_in(hpet, HPET_LEGACY_PIT_INT);

            rtc_irq = qdev_get_gpio_in(hpet, HPET_LEGACY_RTC_INT);

        }

    }

    *rtc_state = rtc_init(isa_bus, 2000, rtc_irq);



    qemu_register_boot_set(pc_boot_set, *rtc_state);



    if (kvm_irqchip_in_kernel()) {

        pit = kvm_pit_init(isa_bus, 0x40);

    } else {

        pit = pit_init(isa_bus, 0x40, pit_isa_irq, pit_alt_irq);

    }

    if (hpet) {

        /* connect PIT to output control line of the HPET */

        qdev_connect_gpio_out(hpet, 0, qdev_get_gpio_in(&pit->qdev, 0));

    }

    pcspk_init(isa_bus, pit);



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_isa_init(isa_bus, i, serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(isa_bus, i, parallel_hds[i]);

        }

    }



    a20_line = qemu_allocate_irqs(handle_a20_line_change, first_cpu, 2);

    i8042 = isa_create_simple(isa_bus, "i8042");

    i8042_setup_a20_line(i8042, &a20_line[0]);

    if (!no_vmport) {

        vmport_init(isa_bus);

        vmmouse = isa_try_create(isa_bus, "vmmouse");

    } else {

        vmmouse = NULL;

    }

    if (vmmouse) {

        qdev_prop_set_ptr(&vmmouse->qdev, "ps2_mouse", i8042);

        qdev_init_nofail(&vmmouse->qdev);

    }

    port92 = isa_create_simple(isa_bus, "port92");

    port92_init(port92, &a20_line[1]);



    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    DMA_init(0, cpu_exit_irq);



    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    *floppy = fdctrl_init_isa(isa_bus, fd);

}
