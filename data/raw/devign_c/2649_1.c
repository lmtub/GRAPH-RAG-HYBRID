static void pci_ivshmem_realize(PCIDevice *dev, Error **errp)

{

    IVShmemState *s = IVSHMEM(dev);

    Error *err = NULL;

    uint8_t *pci_conf;

    uint8_t attr = PCI_BASE_ADDRESS_SPACE_MEMORY |

        PCI_BASE_ADDRESS_MEM_PREFETCH;



    if (!!s->server_chr + !!s->shmobj + !!s->hostmem != 1) {

        error_setg(errp,

                   "You must specify either 'shm', 'chardev' or 'x-memdev'");

        return;

    }



    if (s->hostmem) {

        MemoryRegion *mr;



        if (s->sizearg) {

            g_warning("size argument ignored with hostmem");

        }



        mr = host_memory_backend_get_memory(s->hostmem, &error_abort);

        s->ivshmem_size = memory_region_size(mr);

    } else if (s->sizearg == NULL) {

        s->ivshmem_size = 4 << 20; /* 4 MB default */

    } else {

        char *end;

        int64_t size = qemu_strtosz(s->sizearg, &end);

        if (size < 0 || *end != '\0' || !is_power_of_2(size)) {

            error_setg(errp, "Invalid size %s", s->sizearg);

            return;

        }

        s->ivshmem_size = size;

    }



    /* IRQFD requires MSI */

    if (ivshmem_has_feature(s, IVSHMEM_IOEVENTFD) &&

        !ivshmem_has_feature(s, IVSHMEM_MSI)) {

        error_setg(errp, "ioeventfd/irqfd requires MSI");

        return;

    }



    /* check that role is reasonable */

    if (s->role) {

        if (strncmp(s->role, "peer", 5) == 0) {

            s->role_val = IVSHMEM_PEER;

        } else if (strncmp(s->role, "master", 7) == 0) {

            s->role_val = IVSHMEM_MASTER;

        } else {

            error_setg(errp, "'role' must be 'peer' or 'master'");

            return;

        }

    } else {

        s->role_val = IVSHMEM_MASTER; /* default */

    }



    pci_conf = dev->config;

    pci_conf[PCI_COMMAND] = PCI_COMMAND_IO | PCI_COMMAND_MEMORY;



    /*

     * Note: we don't use INTx with IVSHMEM_MSI at all, so this is a

     * bald-faced lie then.  But it's a backwards compatible lie.

     */

    pci_config_set_interrupt_pin(pci_conf, 1);



    memory_region_init_io(&s->ivshmem_mmio, OBJECT(s), &ivshmem_mmio_ops, s,

                          "ivshmem-mmio", IVSHMEM_REG_BAR_SIZE);



    /* region for registers*/

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY,

                     &s->ivshmem_mmio);



    memory_region_init(&s->bar, OBJECT(s), "ivshmem-bar2-container", s->ivshmem_size);

    if (s->ivshmem_64bit) {

        attr |= PCI_BASE_ADDRESS_MEM_TYPE_64;

    }



    if (s->hostmem != NULL) {

        MemoryRegion *mr;



        IVSHMEM_DPRINTF("using hostmem\n");



        mr = host_memory_backend_get_memory(MEMORY_BACKEND(s->hostmem),

                                            &error_abort);

        vmstate_register_ram(mr, DEVICE(s));

        memory_region_add_subregion(&s->bar, 0, mr);

        pci_register_bar(PCI_DEVICE(s), 2, attr, &s->bar);

    } else if (s->server_chr != NULL) {

        /* FIXME do not rely on what chr drivers put into filename */

        if (strncmp(s->server_chr->filename, "unix:", 5)) {

            error_setg(errp, "chardev is not a unix client socket");

            return;

        }



        /* if we get a UNIX socket as the parameter we will talk

         * to the ivshmem server to receive the memory region */



        IVSHMEM_DPRINTF("using shared memory server (socket = %s)\n",

                        s->server_chr->filename);



        if (ivshmem_setup_interrupts(s) < 0) {

            error_setg(errp, "failed to initialize interrupts");

            return;

        }



        /* we allocate enough space for 16 peers and grow as needed */

        resize_peers(s, 16);

        s->vm_id = -1;



        pci_register_bar(dev, 2, attr, &s->bar);



        qemu_chr_add_handlers(s->server_chr, ivshmem_can_receive,

                              ivshmem_check_version, NULL, s);

    } else {

        /* just map the file immediately, we're not using a server */

        int fd;



        IVSHMEM_DPRINTF("using shm_open (shm object = %s)\n", s->shmobj);



        /* try opening with O_EXCL and if it succeeds zero the memory

         * by truncating to 0 */

        if ((fd = shm_open(s->shmobj, O_CREAT|O_RDWR|O_EXCL,

                        S_IRWXU|S_IRWXG|S_IRWXO)) > 0) {

           /* truncate file to length PCI device's memory */

            if (ftruncate(fd, s->ivshmem_size) != 0) {

                error_report("could not truncate shared file");

            }



        } else if ((fd = shm_open(s->shmobj, O_CREAT|O_RDWR,

                        S_IRWXU|S_IRWXG|S_IRWXO)) < 0) {

            error_setg(errp, "could not open shared file");

            return;

        }



        if (check_shm_size(s, fd, errp) == -1) {

            return;

        }



        create_shared_memory_BAR(s, fd, attr, &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

    }



    fifo8_create(&s->incoming_fifo, sizeof(int64_t));



    if (s->role_val == IVSHMEM_PEER) {

        error_setg(&s->migration_blocker,

                   "Migration is disabled when using feature 'peer mode' in device 'ivshmem'");

        migrate_add_blocker(s->migration_blocker);

    }

}
