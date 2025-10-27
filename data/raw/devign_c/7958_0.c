static void ohci_sof(OHCIState *ohci)

{

    ohci->sof_time = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    timer_mod(ohci->eof_timer, ohci->sof_time + usb_frame_time);

    ohci_set_interrupt(ohci, OHCI_INTR_SF);

}
