static void ehci_mem_writel(void *ptr, target_phys_addr_t addr, uint32_t val)
{
    EHCIState *s = ptr;
    uint32_t *mmio = (uint32_t *)(&s->mmio[addr]);
    uint32_t old = *mmio;
    int i;
    trace_usb_ehci_mmio_writel(addr, addr2str(addr), val);
    /* Only aligned reads are allowed on OHCI */
    if (addr & 3) {
        fprintf(stderr, "usb-ehci: Mis-aligned write to addr 0x"
                TARGET_FMT_plx "\n", addr);
        return;
    if (addr >= PORTSC && addr < PORTSC + 4 * NB_PORTS) {
        handle_port_status_write(s, (addr-PORTSC)/4, val);
        trace_usb_ehci_mmio_change(addr, addr2str(addr), *mmio, old);
        return;
    if (addr < OPREGBASE) {
        fprintf(stderr, "usb-ehci: write attempt to read-only register"
                TARGET_FMT_plx "\n", addr);
        return;
    /* Do any register specific pre-write processing here.  */
    switch(addr) {
    case USBCMD:
        if (val & USBCMD_HCRESET) {
            ehci_reset(s);
            val = s->usbcmd;
            break;
        /* not supporting dynamic frame list size at the moment */
        if ((val & USBCMD_FLS) && !(s->usbcmd & USBCMD_FLS)) {
            fprintf(stderr, "attempt to set frame list size -- value %d\n",
                    val & USBCMD_FLS);
            val &= ~USBCMD_FLS;
        if (((USBCMD_RUNSTOP | USBCMD_PSE | USBCMD_ASE) & val) !=
            ((USBCMD_RUNSTOP | USBCMD_PSE | USBCMD_ASE) & s->usbcmd)) {
            if (s->pstate == EST_INACTIVE) {
                SET_LAST_RUN_CLOCK(s);
            s->usbcmd = val; /* Set usbcmd for ehci_update_halt() */
            ehci_update_halt(s);
            qemu_mod_timer(s->frame_timer, qemu_get_clock_ns(vm_clock));
        break;
    case USBSTS:
        val &= USBSTS_RO_MASK;              // bits 6 through 31 are RO
        ehci_clear_usbsts(s, val);          // bits 0 through 5 are R/WC
        val = s->usbsts;
        ehci_update_irq(s);
        break;
    case USBINTR:
        val &= USBINTR_MASK;
        break;
    case FRINDEX:
        val &= 0x00003ff8; /* frindex is 14bits and always a multiple of 8 */
        break;
    case CONFIGFLAG:
        val &= 0x1;
        if (val) {
            for(i = 0; i < NB_PORTS; i++)
                handle_port_owner_write(s, i, 0);
        break;
    case PERIODICLISTBASE:
        if (ehci_periodic_enabled(s)) {
            fprintf(stderr,
              "ehci: PERIODIC list base register set while periodic schedule\n"
              "      is enabled and HC is enabled\n");
        break;
    case ASYNCLISTADDR:
        if (ehci_async_enabled(s)) {
            fprintf(stderr,
              "ehci: ASYNC list address register set while async schedule\n"
              "      is enabled and HC is enabled\n");
        break;
    *mmio = val;
    trace_usb_ehci_mmio_change(addr, addr2str(addr), *mmio, old);