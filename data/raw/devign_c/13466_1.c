static inline int ohci_put_hcca(OHCIState *ohci,

                                uint32_t addr, struct ohci_hcca *hcca)

{

    cpu_physical_memory_write(addr + ohci->localmem_base, hcca, sizeof(*hcca));

    return 1;

}
