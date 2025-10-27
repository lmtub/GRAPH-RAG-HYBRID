void pc_basic_device_init(qemu_irq *isa_irq,

                          ISADevice **rtc_state)

{

    int i;

    DriveInfo *fd[MAX_FD];

    PITState *pit;

    qemu_irq rtc_irq = NULL;

    qemu_irq *a20_line;

    ISADevice *i8042, *port92, *vmmouse;

    qemu_irq *cpu_exit_irq;



    register_ioport_write(0x80, 1, 1, ioport80_write, NULL);



    register_ioport_write(0xf0, 1, 1, ioportF0_write, NULL);



    if (!no_hpet) {

        DeviceState *hpet = sysbus_try_create_simple("hpet", HPET_BASE, NULL);



        if (hpet) {

            for (i = 0; i < 24; i++) {

                sysbus_connect_irq(sysbus_from_qdev(hpet), i, isa_irq[i]);

            }

            rtc_irq = qdev_get_gpio_in(hpet, 0);

        }

    }

    *rtc_state = rtc_init(2000, rtc_irq);



    qemu_register_boot_set(pc_boot_set, *rtc_state);



    pit = pit_init(0x40, isa_reserve_irq(0));

    pcspk_init(pit);



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_isa_init(i, serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(i, parallel_hds[i]);

        }

    }



    a20_line = qemu_allocate_irqs(handle_a20_line_change, first_cpu, 2);

    i8042 = isa_create_simple("i8042");

    i8042_setup_a20_line(i8042, &a20_line[0]);

    vmport_init();

    vmmouse = isa_try_create("vmmouse");

    if (vmmouse) {

        qdev_prop_set_ptr(&vmmouse->qdev, "ps2_mouse", i8042);

    }

    port92 = isa_create_simple("port92");

    port92_init(port92, &a20_line[1]);



    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    DMA_init(0, cpu_exit_irq);



    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    fdctrl_init_isa(fd);

}
