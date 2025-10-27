static void ivshmem_common_realize(PCIDevice *dev, Error **errp)

{

    IVShmemState *s = IVSHMEM_COMMON(dev);

    Error *err = NULL;

    uint8_t *pci_conf;

    uint8_t attr = PCI_BASE_ADDRESS_SPACE_MEMORY |

        PCI_BASE_ADDRESS_MEM_PREFETCH;

    Error *local_err = NULL;



    /* IRQFD requires MSI */

    if (ivshmem_has_feature(s, IVSHMEM_IOEVENTFD) &&

        !ivshmem_has_feature(s, IVSHMEM_MSI)) {

        error_setg(errp, "ioeventfd/irqfd requires MSI");

        return;

    }



    pci_conf = dev->config;

    pci_conf[PCI_COMMAND] = PCI_COMMAND_IO | PCI_COMMAND_MEMORY;



    memory_region_init_io(&s->ivshmem_mmio, OBJECT(s), &ivshmem_mmio_ops, s,

                          "ivshmem-mmio", IVSHMEM_REG_BAR_SIZE);



    /* region for registers*/

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY,

                     &s->ivshmem_mmio);



    if (s->not_legacy_32bit) {

        attr |= PCI_BASE_ADDRESS_MEM_TYPE_64;

    }



    if (s->hostmem != NULL) {

        IVSHMEM_DPRINTF("using hostmem\n");



        s->ivshmem_bar2 = host_memory_backend_get_memory(s->hostmem,

                                                         &error_abort);

    } else {

        Chardev *chr = qemu_chr_fe_get_driver(&s->server_chr);

        assert(chr);



        IVSHMEM_DPRINTF("using shared memory server (socket = %s)\n",

                        chr->filename);



        /* we allocate enough space for 16 peers and grow as needed */

        resize_peers(s, 16);



        /*

         * Receive setup messages from server synchronously.

         * Older versions did it asynchronously, but that creates a

         * number of entertaining race conditions.

         */

        ivshmem_recv_setup(s, &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }



        if (s->master == ON_OFF_AUTO_ON && s->vm_id != 0) {

            error_setg(errp,

                       "master must connect to the server before any peers");

            return;

        }



        qemu_chr_fe_set_handlers(&s->server_chr, ivshmem_can_receive,

                                 ivshmem_read, NULL, s, NULL, true);



        if (ivshmem_setup_interrupts(s) < 0) {

            error_setg(errp, "failed to initialize interrupts");

            return;

        }

    }



    if (s->master == ON_OFF_AUTO_AUTO) {

        s->master = s->vm_id == 0 ? ON_OFF_AUTO_ON : ON_OFF_AUTO_OFF;

    }



    if (!ivshmem_is_master(s)) {

        error_setg(&s->migration_blocker,

                   "Migration is disabled when using feature 'peer mode' in device 'ivshmem'");

        migrate_add_blocker(s->migration_blocker, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            error_free(s->migration_blocker);

            return;

        }

    }



    vmstate_register_ram(s->ivshmem_bar2, DEVICE(s));

    pci_register_bar(PCI_DEVICE(s), 2, attr, s->ivshmem_bar2);

}
