static void ehci_advance_periodic_state(EHCIState *ehci)

{

    uint32_t entry;

    uint32_t list;

    const int async = 0;



    // 4.6



    switch(ehci_get_state(ehci, async)) {

    case EST_INACTIVE:

        if ( !(ehci->frindex & 7) && (ehci->usbcmd & USBCMD_PSE)) {

            ehci_set_usbsts(ehci, USBSTS_PSS);

            ehci_set_state(ehci, async, EST_ACTIVE);

            // No break, fall through to ACTIVE

        } else

            break;



    case EST_ACTIVE:

        if ( !(ehci->frindex & 7) && !(ehci->usbcmd & USBCMD_PSE)) {

            ehci_clear_usbsts(ehci, USBSTS_PSS);

            ehci_set_state(ehci, async, EST_INACTIVE);

            break;

        }



        list = ehci->periodiclistbase & 0xfffff000;

        /* check that register has been set */

        if (list == 0) {

            break;

        }

        list |= ((ehci->frindex & 0x1ff8) >> 1);



        pci_dma_read(&ehci->dev, list, &entry, sizeof entry);

        entry = le32_to_cpu(entry);



        DPRINTF("PERIODIC state adv fr=%d.  [%08X] -> %08X\n",

                ehci->frindex / 8, list, entry);

        ehci_set_fetch_addr(ehci, async,entry);

        ehci_set_state(ehci, async, EST_FETCHENTRY);

        ehci_advance_state(ehci, async);

        ehci_queues_rip_unused(ehci, async);

        break;



    default:

        /* this should only be due to a developer mistake */

        fprintf(stderr, "ehci: Bad periodic state %d. "

                "Resetting to active\n", ehci->pstate);

        assert(0);

    }

}
