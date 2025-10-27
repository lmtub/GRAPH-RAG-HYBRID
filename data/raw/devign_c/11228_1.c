static uint32_t ehci_mem_readl(void *ptr, target_phys_addr_t addr)

{

    EHCIState *s = ptr;

    uint32_t val;



    val = s->mmio[addr] | (s->mmio[addr+1] << 8) |

          (s->mmio[addr+2] << 16) | (s->mmio[addr+3] << 24);



    trace_usb_ehci_mmio_readl(addr, addr2str(addr), val);

    return val;

}
