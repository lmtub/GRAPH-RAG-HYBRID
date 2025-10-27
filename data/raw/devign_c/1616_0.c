static void pci_init_mask_bridge(PCIDevice *d)

{

    /* PCI_PRIMARY_BUS, PCI_SECONDARY_BUS, PCI_SUBORDINATE_BUS and

       PCI_SEC_LETENCY_TIMER */

    memset(d->wmask + PCI_PRIMARY_BUS, 0xff, 4);



    /* base and limit */

    d->wmask[PCI_IO_BASE] = PCI_IO_RANGE_MASK & 0xff;

    d->wmask[PCI_IO_LIMIT] = PCI_IO_RANGE_MASK & 0xff;

    pci_set_word(d->wmask + PCI_MEMORY_BASE,

                 PCI_MEMORY_RANGE_MASK & 0xffff);

    pci_set_word(d->wmask + PCI_MEMORY_LIMIT,

                 PCI_MEMORY_RANGE_MASK & 0xffff);

    pci_set_word(d->wmask + PCI_PREF_MEMORY_BASE,

                 PCI_PREF_RANGE_MASK & 0xffff);

    pci_set_word(d->wmask + PCI_PREF_MEMORY_LIMIT,

                 PCI_PREF_RANGE_MASK & 0xffff);



    /* PCI_PREF_BASE_UPPER32 and PCI_PREF_LIMIT_UPPER32 */

    memset(d->wmask + PCI_PREF_BASE_UPPER32, 0xff, 8);



    /* Supported memory and i/o types */

    d->config[PCI_IO_BASE] |= PCI_IO_RANGE_TYPE_16;

    d->config[PCI_IO_LIMIT] |= PCI_IO_RANGE_TYPE_16;

    pci_word_test_and_set_mask(d->config + PCI_PREF_MEMORY_BASE,

                               PCI_PREF_RANGE_TYPE_64);

    pci_word_test_and_set_mask(d->config + PCI_PREF_MEMORY_LIMIT,

                               PCI_PREF_RANGE_TYPE_64);



/* TODO: add this define to pci_regs.h in linux and then in qemu. */

#define  PCI_BRIDGE_CTL_VGA_16BIT	0x10	/* VGA 16-bit decode */

#define  PCI_BRIDGE_CTL_DISCARD		0x100	/* Primary discard timer */

#define  PCI_BRIDGE_CTL_SEC_DISCARD	0x200	/* Secondary discard timer */

#define  PCI_BRIDGE_CTL_DISCARD_STATUS	0x400	/* Discard timer status */

#define  PCI_BRIDGE_CTL_DISCARD_SERR	0x800	/* Discard timer SERR# enable */

/*

 * TODO: Bridges default to 10-bit VGA decoding but we currently only

 * implement 16-bit decoding (no alias support).

 */

    pci_set_word(d->wmask + PCI_BRIDGE_CONTROL,

                 PCI_BRIDGE_CTL_PARITY |

                 PCI_BRIDGE_CTL_SERR |

                 PCI_BRIDGE_CTL_ISA |

                 PCI_BRIDGE_CTL_VGA |

                 PCI_BRIDGE_CTL_VGA_16BIT |

                 PCI_BRIDGE_CTL_MASTER_ABORT |

                 PCI_BRIDGE_CTL_BUS_RESET |

                 PCI_BRIDGE_CTL_FAST_BACK |

                 PCI_BRIDGE_CTL_DISCARD |

                 PCI_BRIDGE_CTL_SEC_DISCARD |

                 PCI_BRIDGE_CTL_DISCARD_SERR);

    /* Below does not do anything as we never set this bit, put here for

     * completeness. */

    pci_set_word(d->w1cmask + PCI_BRIDGE_CONTROL,

                 PCI_BRIDGE_CTL_DISCARD_STATUS);

    d->cmask[PCI_IO_BASE] |= PCI_IO_RANGE_TYPE_MASK;

    d->cmask[PCI_IO_LIMIT] |= PCI_IO_RANGE_TYPE_MASK;

    pci_word_test_and_set_mask(d->cmask + PCI_PREF_MEMORY_BASE,

                               PCI_PREF_RANGE_TYPE_MASK);

    pci_word_test_and_set_mask(d->cmask + PCI_PREF_MEMORY_LIMIT,

                               PCI_PREF_RANGE_TYPE_MASK);

}
