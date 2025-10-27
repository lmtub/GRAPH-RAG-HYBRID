static void apb_pci_config_write(void *opaque, target_phys_addr_t addr,

                                 uint64_t val, unsigned size)

{

    APBState *s = opaque;



    val = qemu_bswap_len(val, size);

    APB_DPRINTF("%s: addr " TARGET_FMT_lx " val %" PRIx64 "\n", __func__, addr, val);

    pci_data_write(s->bus, addr, val, size);

}
