qemu_irq *openpic_init (PCIBus *bus, int *pmem_index, int nb_cpus,

                        qemu_irq **irqs, qemu_irq irq_out)

{

    openpic_t *opp;

    uint8_t *pci_conf;

    int i, m;



    /* XXX: for now, only one CPU is supported */

    if (nb_cpus != 1)

        return NULL;

    if (bus) {

        opp = (openpic_t *)pci_register_device(bus, "OpenPIC", sizeof(openpic_t),

                                               -1, NULL, NULL);

        if (opp == NULL)

            return NULL;

        pci_conf = opp->pci_dev.config;

        pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_IBM);

        pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_IBM_OPENPIC2);

        pci_config_set_class(pci_conf, PCI_CLASS_SYSTEM_OTHER); // FIXME?

        pci_conf[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_NORMAL; // header_type

        pci_conf[0x3d] = 0x00; // no interrupt pin



        /* Register I/O spaces */

        pci_register_bar((PCIDevice *)opp, 0, 0x40000,

                               PCI_BASE_ADDRESS_SPACE_MEMORY, &openpic_map);

    } else {

        opp = qemu_mallocz(sizeof(openpic_t));

    }

    opp->mem_index = cpu_register_io_memory(openpic_read,

                                            openpic_write, opp);



    //    isu_base &= 0xFFFC0000;

    opp->nb_cpus = nb_cpus;

    opp->max_irq = OPENPIC_MAX_IRQ;

    opp->irq_ipi0 = OPENPIC_IRQ_IPI0;

    opp->irq_tim0 = OPENPIC_IRQ_TIM0;

    /* Set IRQ types */

    for (i = 0; i < OPENPIC_EXT_IRQ; i++) {

        opp->src[i].type = IRQ_EXTERNAL;

    }

    for (; i < OPENPIC_IRQ_TIM0; i++) {

        opp->src[i].type = IRQ_SPECIAL;

    }

#if MAX_IPI > 0

    m = OPENPIC_IRQ_IPI0;

#else

    m = OPENPIC_IRQ_DBL0;

#endif

    for (; i < m; i++) {

        opp->src[i].type = IRQ_TIMER;

    }

    for (; i < OPENPIC_MAX_IRQ; i++) {

        opp->src[i].type = IRQ_INTERNAL;

    }

    for (i = 0; i < nb_cpus; i++)

        opp->dst[i].irqs = irqs[i];

    opp->irq_out = irq_out;

    opp->need_swap = 1;



    register_savevm("openpic", 0, 2, openpic_save, openpic_load, opp);

    qemu_register_reset(openpic_reset, opp);



    opp->irq_raise = openpic_irq_raise;

    opp->reset = openpic_reset;



    if (pmem_index)

        *pmem_index = opp->mem_index;



    return qemu_allocate_irqs(openpic_set_irq, opp, opp->max_irq);

}
